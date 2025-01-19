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
        results_df.to_csv(Path(f"{BASE_PATH}/Data/SVM_results/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}.csv"), index=False)

if __name__ == "__main__":
    optimise_SVMs()