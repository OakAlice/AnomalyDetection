from MainScript import BASE_PATH, DATASET_NAME # BEHAVIOUR_SET, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from skopt.space import Real, Categorical
from skopt.optimizer import Optimizer
from skopt.utils import use_named_args
import pandas as pd
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import os

def save_results(results, dataset_name, training_set, model_type, base_path, behaviour_set, thresholding):
    """Save optimization results to CSV"""
    try:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df['behaviour'] = results_df.index
        
        # Add context information to each row
        results_df['dataset_name'] = dataset_name
        results_df['training_set'] = training_set
        results_df['model_type'] = model_type
        
        columns = ['dataset_name', 'training_set', 'model_type', 
                  'behaviour', 'kernel', 'C', 'gamma', 'best_auc', 'elapsed_time']
        results_df = results_df[columns] if len(results_df.columns) > 0 else pd.DataFrame(columns=columns)
        
        if model_type.lower() == 'multi':
            if thresholding is not None:
                output_path = Path(f"{base_path}/Output/Tuning/{dataset_name}_{training_set}_{model_type}_{behaviour_set}_threshold_optimisation_results.csv")
            else:
                output_path = Path(f"{base_path}/Output/Tuning/{dataset_name}_{training_set}_{model_type}_{behaviour_set}_NOthreshold_optimisation_results.csv")
        else:
            output_path = Path(f"{base_path}/Output/Tuning/{dataset_name}_{training_set}_{model_type}_optimisation_results.csv")
        

        print(f"saved to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def optimize_svm(X, y, groups, thresholding):
    """Optimize SVM hyperparameters using Bayesian optimization"""
    space = [
        Real(0.01, 1000, name='C', prior='log-uniform'),
        Real(0.001, 100, name='gamma', prior='log-uniform'),
        Categorical(['linear', 'rbf', 'poly'], name='kernel')
    ]
    
    @use_named_args(space)
    def objective(**params):
        svm = SVC(probability=True, class_weight='balanced', 
                  cache_size=1000, **params)
        
        cv = GroupKFold(n_splits=3)
        scores = []
        
        # Convert to numpy arrays if they aren't already
        X_array = np.array(X)
        y_array = np.array(y)
        groups_array = np.array(groups)
        
        for train_idx, test_idx in cv.split(X_array, y_array, groups_array):
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            
            svm.fit(X_train, y_train)
            y_pred_proba = svm.predict_proba(X_test)
            
            # Apply thresholding if specified
            if thresholding is not None:
                if len(np.unique(y)) == 2:  # Binary case
                    # Create mask for predictions above threshold
                    mask = y_pred_proba[:, 1] >= thresholding
                    if np.sum(mask) > 0 and len(np.unique(y_test[mask])) > 1:
                        y_test_filtered = y_test[mask]
                        y_pred_proba_filtered = y_pred_proba[mask]
                        score = roc_auc_score(y_test_filtered, y_pred_proba_filtered[:, 1])
                    else:
                        score = 0
                else:  # Multiclass case
                    # Get predictions where max probability is above threshold
                    max_probs = np.max(y_pred_proba, axis=1)
                    mask = max_probs >= thresholding
                    if np.sum(mask) > 0 and len(np.unique(y_test[mask])) > 1:
                        y_test_filtered = y_test[mask]
                        y_pred_proba_filtered = y_pred_proba[mask]
                        score = roc_auc_score(y_test_filtered, y_pred_proba_filtered, multi_class='ovr')
                    else:
                        score = 0
            else:
                # Original scoring logic
                if len(np.unique(y)) > 2:
                    score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                else:
                    score = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            scores.append(score)
            
        return -np.mean(scores)  # Negative because we want to maximize
    
    optimizer = Optimizer(space)
    best_score = float('inf')
    best_params = None
    n_iterations = 5
    
    for i in range(n_iterations):
        print(f"\nIteration {i+1}/{n_iterations}")
        
        next_x = optimizer.ask()
        score = objective(next_x)
        optimizer.tell(next_x, score)
        
        if score < best_score:
            best_score = score
            best_params = dict(zip(['C', 'gamma', 'kernel'], next_x))
            print(f"New best score: {-best_score:.4f}")
            print(f"New best params: {best_params}")
        
        print(f"Current score: {-score:.4f}")
        print(f"Current params: {dict(zip(['C', 'gamma', 'kernel'], next_x))}")
    
    return best_params, -best_score

def main(base_path, dataset_name, training_set, model_type, target_activities, behaviour_set, thresholding):
    """Main optimization function"""
    try:
        optimization_results = {}
        scaler = StandardScaler()

        # Handle multiclass case
        if model_type.lower() == 'multi':
            df = pd.read_csv(Path(f"{base_path}/Data/Split_data/{dataset_name}_{training_set}_{model_type}_{behaviour_set}.csv"))
            df = df.groupby(['Activity', 'ID']).head(500)
            
            X = scaler.fit_transform(df.drop(['Activity', 'ID'], axis=1))
            y = df['Activity']
            groups = df['ID']
            
            print("\nProcessing multiclass optimization...")
            print(f"Total samples: {len(y)}")
            print(f"Unique groups: {len(np.unique(groups))}")
            
            start_time = time.time()
            best_params, best_score = optimize_svm(X, y, groups, thresholding)
            elapsed_time = time.time() - start_time
            
            optimization_results[behaviour_set] = {
                'kernel': best_params['kernel'],
                'C': best_params['C'],
                'gamma': best_params['gamma'],
                'best_auc': best_score,
                'elapsed_time': elapsed_time
            }

        # Handle binary case
        else:
            for behaviour in target_activities:
                try:
                    print(f"\nProcessing {behaviour}...")
                    df = pd.read_csv(Path(f"{base_path}/Data/Split_data/{dataset_name}_{training_set}_{model_type}_{behaviour}.csv"))
                    df = df.groupby(['Activity', 'ID']).head(100)
                    
                    X = scaler.fit_transform(df.drop(['Activity', 'ID'], axis=1))
                    y = np.where(df['Activity'] == behaviour, 1, 0)
                    groups = df['ID']
                    
                    print(f"Total samples: {len(y)}")
                    print(f"Unique groups: {len(np.unique(groups))}")
                    print(f"Class distribution: {np.bincount(y)}")
                    
                    start_time = time.time()
                    best_params, best_score = optimize_svm(X, y, groups, thresholding)
                    elapsed_time = time.time() - start_time
                    
                    optimization_results[behaviour] = {
                        'kernel': best_params['kernel'],
                        'C': best_params['C'],
                        'gamma': best_params['gamma'],
                        'best_auc': best_score,
                        'elapsed_time': elapsed_time
                    }
                    
                except Exception as e:
                    print(f"Error processing {behaviour}: {str(e)}")
                    continue
        
        save_results(optimization_results, dataset_name, training_set, model_type, base_path, behaviour_set, thresholding)
        print("\nOptimization completed successfully!")
        
    except Exception as e:
        print(f"Fatal error in main function: {str(e)}")
        raise

def generate_learning_curves(BASE_PATH, DATASET_NAME, TRAINING_SET, TARGET_ACTIVITIES):
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
        output_dir = os.path.join(BASE_PATH, 'Output', 'Tuning')
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

def append_files(BASE_PATH):
    path = f"{BASE_PATH}/Output/Tuning"
    
    # Get all CSV files in the directory
    all_files = [f for f in os.listdir(path) if f.endswith('_optimisation_results.csv')]
    
    if not all_files:
        print("No optimization results files found.")
        return
    
    # Read and combine all CSV files
    dfs = []
    for filename in all_files:
        file_path = os.path.join(path, filename)
        try:
            df = pd.read_csv(file_path)
            # Check if 'threshold' appears anywhere in the filename (case-insensitive)
            if 'threshold' in filename.lower() and 'nothreshold' not in filename.lower():
                df['thresholding'] = True
            else:
                df['thresholding'] = False
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")

            continue
    
    if not dfs:
        print("No valid files could be read.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # modify the column types
    combined_df['C'] = combined_df['C'].astype(float)
    combined_df['gamma'] = combined_df['gamma'].astype(float)
    combined_df['best_auc'] = combined_df['best_auc'].astype(float)
    combined_df['elapsed_time'] = combined_df['elapsed_time'].astype(float)
    
    # Save combined results
    output_path = os.path.join(path, 'Combined_optimisation_results.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined results saved to: {output_path}")

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)

