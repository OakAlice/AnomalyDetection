from MainScript import BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES, TRAINING_SET, MODEL_TYPE
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

class SVMTuner:
    """
    Tuning SVM hyperparameters using Bayesian optimization.
    Supports one-class, binary, and multi-class classification within same function.
    """
    
    def __init__(self, svm_type: str = 'binary', n_iter: int = 25):

        self.svm_type = svm_type
        self.n_iter = n_iter
        self.best_params_ = None
        self.best_score_ = None
        
    def get_search_space(self) -> Dict:
        """
        Creates bounds based on SVM type.
        """
        base_space = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear']),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform')
        }
        
        if self.svm_type == 'oneclass':
            return {
                'kernel': Categorical(['rbf', 'linear']),
                'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
                'nu': Real(0.01, 0.99, prior='uniform')
            }
        return base_space
    
    def get_scorer(self):
        """
        Returns appropriate scoring metric based on SVM type.
        """
        if self.svm_type == 'binary' or self.svm_type == 'oneclass':
            return make_scorer(f1_score)
        else:
            return make_scorer(f1_score, average='weighted')
    
    def SVM_tune(self, X, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None, cv: int = 5) -> Dict:
        """
        Perform hyperparameter tuning using Bayesian optimization.
        
        Args:
            X: Feature matrix
            y: Target labels (not needed for one-class though)
            groups: IDs for group-based cross-validation
            cv: Number of cross-validation folds
            
        Returns:
            Dict containing best parameters and scores
        """
        # Initialize appropriate SVM type
        if self.svm_type == 'oneclass':
            model = OneClassSVM()
            cv = None  # no cross-validation for this system
        else:
            model = SVC(random_state=42)
            if groups is not None:
                cv = GroupKFold(n_splits=cv)
        
        # Set up Bayesian search
        opt = BayesSearchCV(
            estimator=model,
            search_spaces=self.get_search_space(),
            n_iter=self.n_iter,
            scoring=self.get_scorer(),
            cv=cv if groups is None else cv.split(X, y, groups),
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Fit the model
        if self.svm_type == 'oneclass':
            opt.fit(X)
        else:
            opt.fit(X, y)
        
        self.best_params_ = opt.best_params_
        self.best_score_ = opt.best_score_
        
        # Create a DataFrame with the optimization results
        results_df = pd.DataFrame(opt.cv_results_)
        
        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'optimization_results': results_df,
            'optimizer': opt
        }

def optimise_SVMs():
    """
    Load in the data and run the code for the SVM optimisation
    """
    for behaviour in TARGET_ACTIVITIES:
        df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
        # remove these columns if they still exist
        df = df.drop(columns=['OtherActivity', 'GeneralisedActivity'], errors='ignore')

        if MODEL_TYPE == 'oneclass':
            df = df[df['Activity']== behaviour]
        
        X = df.drop(columns=['Activity'])
        Y = df['Activity']
        groups = df['ID'] 

        # Binary classification print out and results storing
        print(f"\nPerforming Bayesian optimization for {MODEL_TYPE}, {TRAINING_SET}, {behaviour}...")
        binary_tuner = SVMTuner(svm_type={MODEL_TYPE}, n_iter=10) 
        binary_results = binary_tuner.SVM_tune(X, Y, groups=groups)


        print(f"Best parameters: {binary_results['best_params']}")
        print(f"Best score: {binary_results['best_score']:.3f}")

        # create a dataframe to store the results
        results_df = pd.DataFrame(binary_results['optimization_results'])
        # append this data to the existing results and save iteratively
        results_df = pd.concat([results_df, results_df], ignore_index=True)
        results_df.to_csv(Path(f"{BASE_PATH}/Output/Tuning/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}.csv"), index=False)

if __name__ == "__main__":
    optimise_SVMs()



# alternative style
# Train the optimal model

from MainScript import BASE_PATH, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path

def train_test_svm(DATASET_NAME, TRAINING_SET, MODEL_TYPE, BASE_PATH, TARGET_ACTIVITIES, best_params):
    """
    Train and test SVM models based on the specified type and previously identified best parameters
    
    Args:
        dataset_name (str): Name of the dataset
        training_set (str): Training set identifier
        model_type (str): 'oneclass', 'binary', or 'multiclass'
        base_path (str): Base path for data
        target_activities (list): List of target activities
        best_params (dict): Dictionary of best parameters from optimization
    """
    results = {}
    
    for behaviour in TARGET_ACTIVITIES:
        print(f"\nTraining {MODEL_TYPE} SVM for {behaviour}...")
        
        # Load data
        df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
        
        # Prepare features
        X = df.drop(columns=['Activity', 'ID'])
        y = df['Activity']
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        if MODEL_TYPE.lower() == 'oneclass':
            # For one-class, only use normal class for training
            normal_idx = y == behaviour
            X_train = X[normal_idx]
            X_test = X
            y_test = y  # Keep all labels for evaluation
            
            # Train model
            model = OneClassSVM(**best_params[behaviour])
            model.fit(X_train)
            
            # Predict
            y_pred = model.predict(X_test)
            # Convert predictions from 1/-1 to behavior/"Other"
            y_pred = np.where(y_pred == 1, behaviour, "Other")
            
        else:  # binary or multiclass
            # Split maintaining group structure
            groups = df['ID']
            unique_groups = groups.unique()
            train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)
            
            train_idx = groups.isin(train_groups)
            test_idx = groups.isin(test_groups)
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = SVC(**best_params[behaviour])
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
        
        # Evaluate
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        
        # Store results
        results[behaviour] = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': conf_matrix,
            'model': model,
            'scaler': scaler
        }
        
        # Save predictions
        results_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Label': y_pred
        })
        results_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv"),
                         index=False)
    
    return results

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

def main():
    # Load best parameters from CSV
    best_params = load_best_params(Path(f"{BASE_PATH}/Output/Tuning/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_hyperparameters.csv"))
    
    # Train and test binary SVM
    print("\nTraining Binary SVM models...")
    binary_results = train_test_svm(
        DATASET_NAME=DATASET_NAME,
        TRAINING_SET=TRAINING_SET,
        MODEL_TYPE='binary',
        BASE_PATH=BASE_PATH,
        TARGET_ACTIVITIES=TARGET_ACTIVITIES,
        best_params=best_params
    )
    print(binary_results)
    save_results_to_csv(binary_results, DATASET_NAME, TRAINING_SET, 'binary', BASE_PATH)
    
    # Train and test multi-class SVM
    # print("\nTraining Multi-class SVM models...")
    # multiclass_results = train_test_svm(
    #     DATASET_NAME=DATASET_NAME,
    #     TRAINING_SET=TRAINING_SET,
    #     MODEL_TYPE='multiclass',
    #     BASE_PATH=BASE_PATH,
    #     TARGET_ACTIVITIES=TARGET_ACTIVITIES,
    #     best_params=best_params
    # )

if __name__ == "__main__":
    main()

