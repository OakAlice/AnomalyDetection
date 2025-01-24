from MainScript import BASE_PATH, BEHAVIOUR_SET, BEHAVIOUR_SETS, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
import pandas as pd
import numpy as np
from pathlib import Path
from SVMTrainModel import train_svm
import time

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

def get_weighted_auc(y_true, y_pred, sample_weight=None):
    """Calculate weighted average AUC for multiclass"""
    # Convert inputs to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    # Calculate binary AUC for each class
    auc_scores = []
    weights = []
    for cls in classes:
        # Create binary labels
        y_true_binary = np.array(y_true == cls).astype(int)
        y_pred_binary = np.array(y_pred == cls).astype(int)
        
        # Get class weight
        class_samples = np.sum(y_true_binary)
        weights.append(class_samples)
        
        try:
            auc = roc_auc_score(y_true_binary, y_pred_binary)
            auc_scores.append(auc)
        except:
            auc_scores.append(0.0)
    
    # Calculate weighted average
    weights = np.array(weights) / np.sum(weights)
    weighted_auc = np.average(auc_scores, weights=weights)
    return weighted_auc

def save_optimization_results(results_dict, dataset_name, training_set, model_type, base_path):
    """
    Save optimization results to CSV
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    results_df['behaviour'] = results_df.index
    results_df = results_df[['behaviour', 'kernel', 'C', 'gamma', 'best_auc', 'elapsed_time']]
    
    # Save to CSV
    output_path = Path(f"{base_path}/Results/{dataset_name}_{training_set}_{model_type}_optimization_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Optimization results saved to {output_path}")

def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET):
    try:
        print("Starting SVM training and testing process...")
        optimization_results = {}
        
        # Define the parameter search space
        param_space = {
            'kernel': Categorical(['rbf', 'linear', 'poly', 'sigmoid']),
            'C': Real(1e-3, 1e3, prior='log-uniform'),
            'gamma': Real(1e-3, 1e3, prior='log-uniform')
        }

        if MODEL_TYPE.lower() == 'multi':
            df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}.csv"))
            
            # Prepare data
            X = df.drop(['Activity', 'ID'], axis=1)
            y = df['Activity']
            groups = df['ID']  # For group-based CV splitting

            group_kfold = GroupKFold(n_splits=3)

            base_svm = SVC(probability=True, random_state=42)

            # Initialize BayesSearchCV
            print(f"Beginning optimisation for multiclass SVM model")
            opt = BayesSearchCV(
                base_svm,
                param_space,
                n_iter=20,  # Number of parameter settings that are sampled
                cv=group_kfold.split(X, y, groups),
                scoring=get_weighted_auc,
                n_jobs=-1,
                random_state=42,
                verbose=2
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
                print(f"Beginning optimisation for {MODEL_TYPE} {behaviour} SVM model...")
                df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
                
                # Prepare data
                X = df.drop(['Activity', 'ID'], axis=1)
                y = np.where(df['Activity'] == behaviour, behaviour, 'Other') # binary labelling
                groups = df['ID']

                # Initialize GroupKFold
                group_kfold = GroupKFold(n_splits=3)

                # Initialize SVM model
                base_svm = SVC(probability=True, random_state=42)

                # Initialize BayesSearchCV
                opt = BayesSearchCV(
                    base_svm,
                    param_space,
                    n_iter=20,
                    cv=group_kfold.split(X, y, groups),
                    scoring='roc_auc',  # For binary classification
                    n_jobs=-1,
                    random_state=42,
                    verbose=2
                )

                # Fit the optimizer
                start_time = time.time()
                opt.fit(X, y, groups=groups)
                elapsed_time = time.time() - start_time

                optimization_results[behaviour] = {
                    'kernel': opt.best_params_['kernel'],
                    'C': opt.best_params_['C'],
                    'gamma': opt.best_params_['gamma'],
                    'best_auc': opt.best_score_,
                    'elapsed_time': elapsed_time
                }

        print("\nSaving optimization results...")
        save_optimization_results(optimization_results, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BASE_PATH)
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Fatal error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)

