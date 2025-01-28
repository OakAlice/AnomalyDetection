from MainScript import BASE_PATH, BEHAVIOUR_SET, BEHAVIOUR_SETS, ML_METHOD, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import pandas as pd
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import make_scorer

def save_results_to_csv(results, dataset_name, training_set, model_type, base_path):
    """
    Save classification results to CSV files
    """
    for behavior, result in results.items():
        # Convert classification report dict to dataframe
        class_report_df = pd.DataFrame(result['classification_report']).transpose()
        
        # Convert confusion matrix to dataframe
        conf_matrix_df = pd.DataFrame(
            result['confusion_matrix'],
            columns=[f'Predicted_{label}' for label in ['Other', behavior]],
            index=[f'Actual_{label}' for label in ['Other', behavior]]
        )
        
        # Save classification report
        class_report_df.to_csv(
            Path(f"{base_path}/Results/{dataset_name}_{training_set}_{model_type}_{behavior}_classification_report.csv")
        )
        
        # Save confusion matrix
        conf_matrix_df.to_csv(
            Path(f"{base_path}/Results/{dataset_name}_{training_set}_{model_type}_{behavior}_confusion_matrix.csv")
        )

def load_best_params(csv_path: str) -> dict:
    """
    Load best parameters from CSV and convert to format needed for SVM training
    
    Args:
        csv_path: Path to CSV file containing best parameters
    
    Returns:
        Dictionary of best parameters for each behavior
    """
    # Read the CSV file
    params_df = pd.read_csv(csv_path)
    
    # Convert DataFrame to dictionary format needed for SVM
    best_params = {}
    for _, row in params_df.iterrows():
        behavior = row['behaviour']
        best_params[behavior] = {
            'kernel': row['kernel'], 
            'gamma': row['gamma'],
            'C': row['nu']
        }
    
    return best_params

def get_weighted_auc(y_true, y_pred_proba):
    """
    Calculate weighted average AUC for both binary and multiclass classification
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities from model.predict_proba()
        
    Returns:
        Weighted average AUC score
    """
    y_true, y_pred_proba = np.array(y_true), np.array(y_pred_proba)
    
    # Binary classification
    if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
        
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.5
            
        try:
            positive_class = unique_classes[1] if unique_classes[0] == 'Other' else unique_classes[0]
            return roc_auc_score((y_true == positive_class).astype(int), y_pred_proba)
        except ValueError:
            return 0.5
    
    # Multiclass classification
    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.5
        
    auc_scores, weights = [], []
    for i, cls in enumerate(classes):
        y_true_binary = (y_true == cls).astype(int)
        weights.append(np.sum(y_true_binary))
        
        try:
            auc_scores.append(roc_auc_score(y_true_binary, y_pred_proba[:, i]))
        except ValueError:
            auc_scores.append(0.5)
    
    return np.average(auc_scores, weights=np.array(weights)/np.sum(weights)) if auc_scores else 0.5

# Create a scorer function for sklearn
weighted_auc_scorer = make_scorer(
    get_weighted_auc,
    needs_proba=True,
    greater_is_better=True
)

def save_optimization_results(results_dict, dataset_name, training_set, model_type, base_path, ML_METHOD):
    """
    Save optimization results to CSV
    """
    try:
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        results_df['behaviour'] = results_df.index
        
        # Check which columns are actually present
        expected_columns = ['behaviour', 'kernel', 'C', 'gamma', 'best_auc', 'elapsed_time']
        available_columns = [col for col in expected_columns if col in results_df.columns]
        
        # If no results were collected, create empty DataFrame with expected columns
        if len(available_columns) == 0:
            results_df = pd.DataFrame(columns=expected_columns)
        else:
            results_df = results_df[available_columns]
        
        # Save to CSV
        output_path = Path(f"{base_path}/Output/Tuning/{ML_METHOD}/{dataset_name}_{training_set}_{model_type}_optimisation_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        results_df.to_csv(output_path, index=False)
        print(f"Optimization results saved to {output_path}")
        
    except Exception as e:
        print(f"Error saving optimization results: {str(e)}")
        # Create empty results file to prevent future errors
        empty_df = pd.DataFrame(columns=['behaviour', 'kernel', 'C', 'gamma', 'best_auc', 'elapsed_time'])
        output_path = Path(f"{base_path}/Output/Tuning/{ML_METHOD}/{dataset_name}_{training_set}_{model_type}_optimisation_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(output_path, index=False)

def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET):
    try:
        print("Starting SVM optimization process...")
        optimization_results = {}
        scaler = StandardScaler()

        if MODEL_TYPE.lower() == 'multi':
            df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}.csv"))
            
            # downsampling the datase to 1000 samples per class... requires knowing the orignal class per "other" Activity
            # i dont have this information, so, for now, I just have to hope for the best
            df = df.groupby(['Activity', 'ID']).head(1000) 

            # Prepare data
            X = df.drop(['Activity', 'ID'], axis=1)
            X = scaler.fit_transform(X)
            y = df['Activity']
            groups = df['ID']  # For group-based CV splitting

            group_kfold = GroupKFold(n_splits=3)

            unique_classes = y.unique()
            n_samples = len(y)
            class_weights = {}
            
            for cls in unique_classes:
                n_cls = sum(y == cls)
                # Weight is inversely proportional to class frequency
                class_weights[cls] = n_samples / (len(unique_classes) * n_cls)

            base_svm = SVC(
                probability=True,  # Enable probability estimates
                class_weight=class_weights,
                cache_size=1000
            )

            # Define parameter space using proper skopt space objects
            param_space = {
                'kernel': Categorical(['linear', 'rbf', 'poly']),
                'C': Real(0.01, 100, prior='log-uniform', name='C'),
                'gamma': Real(1e-4, 1, prior='log-uniform', name='gamma'),
            }

            # Initialize BayesSearchCV
            opt = BayesSearchCV(
                base_svm,
                param_space,
                n_iter=10,  # Increase this for more thorough optimization
                cv=group_kfold.split(X, y, groups),
                scoring=weighted_auc_scorer,
                n_jobs=-1,
                random_state=42,
                verbose=3
            )

            # Fit the optimizer
            start_time = time.time()
            opt.fit(X, y, groups=groups)
            elapsed_time = time.time() - start_time

            optimization_results[BEHAVIOUR_SET] = {
                'kernel': opt.best_params_['kernel'],
                'C': opt.best_params_['C'],
                'gamma': opt.best_params_['gamma'],
                'best_auc': opt.best_score_,
                'elapsed_time': elapsed_time
            }

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
                    y = np.where(df['Activity'] == behaviour, behaviour, 'Other') # binary labelling
                    groups = df['ID']

                    # Simplified class distribution output
                    unique_classes, class_counts = np.unique(y, return_counts=True)
                    class_dist = dict(zip(unique_classes, class_counts))

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

                    # Add minimum samples check
                    MIN_SAMPLES_PER_CLASS = 10  # Adjust this threshold as needed
                    if any(count < MIN_SAMPLES_PER_CLASS for count in class_counts):
                        print(f"Insufficient samples for {behaviour}. Minimum {MIN_SAMPLES_PER_CLASS} required per class. Skipping...")
                        continue

                    # Custom GroupKFold that ensures both classes in each fold
                    class CustomGroupKFold:
                        def __init__(self, n_splits=3):
                            self.n_splits = n_splits
                        
                        def split(self, X, y, groups):
                            unique_groups = np.unique(groups)
                            n_groups = len(unique_groups)
                            
                            # Shuffle groups
                            rng = np.random.RandomState(42)
                            group_indices = rng.permutation(n_groups)
                            
                            # Create folds ensuring each has both classes
                            fold_size = n_groups // self.n_splits
                            for i in range(self.n_splits):
                                test_groups = unique_groups[group_indices[i * fold_size:(i + 1) * fold_size]]
                                train_groups = unique_groups[~np.isin(unique_groups, test_groups)]
                                
                                test_idx = np.where(np.isin(groups, test_groups))[0]
                                train_idx = np.where(np.isin(groups, train_groups))[0]
                                
                                # Verify both classes exist in both sets
                                train_classes = np.unique(y[train_idx])
                                test_classes = np.unique(y[test_idx])
                                
                                if len(train_classes) < 2 or len(test_classes) < 2:
                                    continue
                                
                                yield train_idx, test_idx

                    # Initialize custom GroupKFold
                    group_kfold = CustomGroupKFold(n_splits=3)

                    n_other = sum(y == 'Other')
                    n_target = sum(y == behaviour)
                    class_weights = {
                        behaviour: n_other / len(y),
                        'Other': n_target / len(y)
                    }

                    # Initialize SVM model # hardcoded with linear kernel
                    base_svm = SVC(
                        probability=True,  # Enable probability estimates
                        class_weight=class_weights,
                        cache_size=1000
                    )

                    # Add counter for NaN results
                    nan_counter = 0
                    max_nan_allowed = 5  # Adjust this threshold as needed

                    def on_step(optim_result):
                        nonlocal nan_counter
                        current_score = optim_result.fun
                        
                        print(f"\nCurrent iteration: {len(optim_result.x_iters)}")
                        if np.isnan(current_score):
                            nan_counter += 1
                            print(f"WARNING: NaN encountered! NaN count: {nan_counter}/{max_nan_allowed}")
                        print(f"Current best score: {current_score}")
                        print(f"Current best params: {optim_result.x}")
                        
                        # Stop optimization if too many NaNs
                        if nan_counter >= max_nan_allowed:
                            print(f"Stopping optimization: Too many NaN results ({nan_counter})")
                            return False
                        return True

                    # Define parameter space using proper skopt space objects
                    param_space = {
                        'kernel': Categorical(['linear', 'rbf', 'poly']),
                        'C': Real(0.01, 100, prior='log-uniform', name='C'),
                        'gamma': Real(1e-4, 1, prior='log-uniform', name='gamma'),
                    }

                    # Initialize BayesSearchCV
                    opt = BayesSearchCV(
                        base_svm,
                        param_space,
                        n_iter=10,  # Increase this for more thorough optimization
                        cv=group_kfold.split(X, y, groups),
                        scoring=weighted_auc_scorer,
                        n_jobs=-1,
                        random_state=42,
                        verbose=3
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
        save_optimization_results(optimization_results, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BASE_PATH, ML_METHOD)
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
        print(f"Error generating learning curves: {str(e)}")
        raise


if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)

    # generate_learning_curves(BASE_PATH, DATASET_NAME, TRAINING_SET, ML_METHOD, TARGET_ACTIVITIES)

