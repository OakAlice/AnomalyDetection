from MainScript import BASE_PATH, BEHAVIOUR_SET, BEHAVIOUR_SETS, ML_METHOD, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import pandas as pd
import numpy as np
from pathlib import Path
import time

def save_results(results, dataset_name, training_set, model_type, base_path, ml_method):
    """Save optimization results to CSV"""
    try:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df['behaviour'] = results_df.index
        
        columns = ['behaviour', 'kernel', 'C', 'gamma', 'best_auc', 'elapsed_time']
        results_df = results_df[columns] if len(results_df.columns) > 0 else pd.DataFrame(columns=columns)
        
        output_path = Path(f"{base_path}/Output/Tuning/{ml_method}/{dataset_name}_{training_set}_{model_type}_optimisation_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main(base_path, dataset_name, training_set, model_type, target_activities, behaviour_set):
    """Main optimization function"""
    try:
        optimization_results = {}
        scaler = StandardScaler()

        # Handle multiclass case
        if model_type.lower() == 'multi':
            df = pd.read_csv(Path(f"{base_path}/Data/Split_data/{dataset_name}_{training_set}_{model_type}_{behaviour_set}.csv"))
            df = df.groupby(['Activity', 'ID']).head(1000)
            
            X = scaler.fit_transform(df.drop(['Activity', 'ID'], axis=1))
            y = df['Activity']
            groups = df['ID']
            
            # Setup SVM
            class_weights = {cls: len(y)/(len(np.unique(y)) * sum(y == cls)) for cls in np.unique(y)}
            svm = SVC(probability=True, class_weight=class_weights, cache_size=1000)
            
            # Use standard ROC AUC for multiclass
            scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
            
            # Optimize
            opt = BayesSearchCV(
                svm,
                {
                    'kernel': Categorical(['linear', 'rbf', 'poly']),
                    'C': Real(0.01, 100, prior='log-uniform'),
                    'gamma': Real(1e-4, 1, prior='log-uniform'),
                },
                n_iter=10,
                cv=GroupKFold(n_splits=3).split(X, y, groups),
                scoring=scorer,
                n_jobs=-1,
                verbose=2
            )
            
            def on_step(optim_result):
                """Callback to monitor optimization progress"""
                n_iter = len(optim_result.x_iters)
                score = optim_result.fun
                print(f"\nIteration {n_iter}:")
                print(f"Current parameters: {dict(zip(opt.optimizer.space.dimension_names, optim_result.x_iters[-1]))}")
                print(f"Current score: {score}")
                return True

            start_time = time.time()
            opt.fit(X, y, groups=groups, callback=[on_step])
            
            optimization_results[behaviour_set] = {
                'kernel': opt.best_params_['kernel'],
                'C': opt.best_params_['C'],
                'gamma': opt.best_params_['gamma'],
                'best_auc': opt.best_score_,
                'elapsed_time': time.time() - start_time
            }

        # Handle binary case
        else:
            for behaviour in TARGET_ACTIVITIES:
                try:
                    print(f"Optimizing {MODEL_TYPE} SVM model for {behaviour}...")
                    df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
                    
                    # downsampling the datase to 1000 samples per class... requires knowing the orignal class per "other" Activity
                    # i dont have this information, so, for now, I just have to hope for the best
                    df = df.groupby(['Activity', 'ID']).head(100) 

                    # Prepare data
                    X = df.drop(['Activity', 'ID'], axis=1)
                    X = scaler.fit_transform(X)
                    y = np.where(df['Activity'] == behaviour, 1, 0)
                    groups = df['ID']

                    # Check group distribution
                    print(f"\nProcessing {behaviour}:")
                    print(f"Total samples: {len(y)}")
                    print(f"Unique groups: {len(np.unique(groups))}")
                    print(f"Class distribution: {np.bincount(y)}")
                    
                    # Ensure minimum samples per class in each group
                    group_class_counts = df.groupby(['ID', 'Activity']).size().unstack(fill_value=0)
                    valid_groups = group_class_counts[(group_class_counts[behaviour] > 0) & 
                                                    (group_class_counts.drop(behaviour, axis=1).sum(axis=1) > 0)].index

                    if len(valid_groups) < 3:  # We need at least 3 groups for 3-fold CV
                        print(f"Not enough groups with both classes for {behaviour}. Skipping...")
                        continue

                    # Filter data to only include valid groups
                    mask = df['ID'].isin(valid_groups)
                    X = X[mask]
                    y = y[mask]
                    groups = groups[mask]

                    print(f"Valid groups after filtering: {len(np.unique(groups))}")
                    print(f"Samples after filtering: {len(y)}")
                    print(f"Class distribution after filtering: {np.bincount(y)}")

                    # Create CV splits to verify
                    group_kfold = GroupKFold(n_splits=3)
                    cv_splits = list(group_kfold.split(X, y, groups))
                    print(f"Number of CV splits: {len(cv_splits)}")
                    
                    for i, (train_idx, test_idx) in enumerate(cv_splits):
                        print(f"\nSplit {i+1}:")
                        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
                        print(f"Train class dist: {np.bincount(y[train_idx])}")
                        print(f"Test class dist: {np.bincount(y[test_idx])}")

                    # Setup SVM with proper CV
                    svm = SVC(probability=True, class_weight='balanced', cache_size=1000)
                    
                    def on_step(optim_result):
                        """Callback to monitor optimization progress"""
                        n_iter = len(optim_result.x_iters)
                        score = optim_result.fun
                        print(f"\nIteration {n_iter}:")
                        print(f"Current parameters: {optim_result.x_iters[-1]}")
                        print(f"Current score: {score}")
                        return True
                    
                    # Optimize with verified CV splits
                    opt = BayesSearchCV(
                        svm,
                        {
                            'kernel': Categorical(['linear', 'rbf', 'poly']),
                            'C': Real(0.01, 100, prior='log-uniform'),
                            'gamma': Real(1e-4, 1, prior='log-uniform'),
                        },
                        n_iter=50,
                        cv=GroupKFold(n_splits=3),
                        scoring='roc_auc',
                        n_jobs=-1,
                        random_state=hash(behaviour) % 1000,
                        verbose=2
                    )

                    start_time = time.time()
                    try:
                        opt.fit(X, y, groups=groups, callback=[on_step])
                        elapsed_time = time.time() - start_time
                        
                        # Store the results for this behavior
                        optimization_results[behaviour] = {
                            'kernel': opt.best_params_['kernel'],
                            'C': opt.best_params_['C'],
                            'gamma': opt.best_params_['gamma'],
                            'best_auc': opt.best_score_,
                            'elapsed_time': elapsed_time
                        }
                        
                        print(f"Best score: {opt.best_score_:.4f}, Best params: {opt.best_params_}")
                        
                    except Exception as e:
                        print(f"Optimization error for {behaviour}: {str(e)}")
                        continue
                    
                except Exception as e:
                    print(f"Error processing {behaviour}: {str(e)}")
                    continue
        save_results(optimization_results, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BASE_PATH, ML_METHOD)
        print("Optimization completed successfully!")
        
    except Exception as e:
        print(f"Fatal error in main function: {str(e)}")
        raise

def generate_learning_curves(BASE_PATH, DATASET_NAME, TRAINING_SET, ML_METHOD, TARGET_ACTIVITIES):
    try:
        print("\nGenerating learning curve...")
        behaviour = TARGET_ACTIVITIES[0] 
        print(f"\nProcessing behaviour: {behaviour}")
        
        print("Loading and preprocessing data...")
        # Load and preprocess data
        df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_all_Binary_{behaviour}.csv"))
        print(f"Loaded dataset with shape: {df.shape}")
                
        # Prepare data
        X = df.drop(['Activity', 'ID'], axis=1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.where(df['Activity'] == behaviour, behaviour, 'Other') # binary labelling
        groups = df['ID']
        
        learning_curve_results = {}
            
        print("Calculating class weights...")
        # Calculate class weights
        n_other = sum(y == 'Other')
        n_target = sum(y == behaviour)
        class_weights = {
            behaviour: n_other / len(y),
            'Other': n_target / len(y)
        }
            
        # Initialize SVM with default parameters
        print("Initializing SVM model...")
        svm = SVC(probability=True, class_weight=class_weights, cache_size=1000)
            
        # Define training sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
            
        print("Generating learning curves (this may take a while)...")
        # Generate learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            svm, X, y,
            cv=GroupKFold(n_splits=2).split(X, y, groups),
            train_sizes=train_sizes,
            scoring='roc_auc',
            n_jobs=-1,
            groups=groups,
            verbose=1  # Added verbose parameter
        )
            
        print("Calculating statistics...")
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
            
        learning_curve_results[behaviour] = {
            'train_sizes': train_sizes.tolist(),  # Convert numpy arrays to lists for JSON serialization
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'test_mean': test_mean.tolist(),
            'test_std': test_std.tolist()
        }
            
        print("Creating and saving plot...")
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, test_mean, label='Cross-validation score')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
            
        plt.xlabel('Training Examples')
        plt.ylabel('ROC AUC Score')
        plt.title(f'Learning Curves for {behaviour}')
        plt.legend(loc='best')
        plt.grid(True)
            
        # Create directory if it doesn't exist
        output_dir = os.path.join(BASE_PATH, 'Output', 'Tuning', ML_METHOD)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plot
        plot_path = os.path.join(output_dir, f'Binary_{behaviour}_learning_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to: {plot_path}")
                
        print("\nLearning curve generated successfully!")
            
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)

