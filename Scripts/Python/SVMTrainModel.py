# Train the optimal model

from MainScript import BASE_PATH, BEHAVIOUR_SET, ML_METHOD, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import numpy as np
from pathlib import Path

def train_svm(df, MODEL_TYPE, kernel, nu, gamma, behaviour):
    """
    Train a single SVM model based on the specified type and previously identified best parameters
    
    Args:
        df (df): training data to build this model
        dataset_name (str): Name of the dataset
        training_set (str): Training set identifier
        model_type (str): 'oneclass', 'binary', or 'multiclass'
        base_path (str): Base path for data
        target_activities (list): List of target activities
        best_params (dict): Dictionary of best parameters from optimization
        behaviour (str): The specific behavior to train for
    
    Returns:
        dict: Dictionary containing the model and scaler
    """
    print(f"\nTraining {MODEL_TYPE} SVM for {behaviour or set}...")
    
    # Prepare features
    X = df.drop(columns=['Activity', 'ID'])
    y = df['Activity']
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Add class_weight parameter to handle imbalance
    if MODEL_TYPE.lower() == 'binary':
        n_other = sum(y == 'Other')
        n_target = sum(y == behaviour)
        class_weights = {
            behaviour: n_other / len(y),
            'Other': n_target / len(y)
        }
    elif MODEL_TYPE.lower() == 'multi':
        # Calculate class weights for all unique classes
        unique_classes = y.unique()
        n_samples = len(y)
        class_weights = {}
        
        for cls in unique_classes:
            n_cls = sum(y == cls)
            # Weight is inversely proportional to class frequency
            class_weights[cls] = n_samples / (len(unique_classes) * n_cls)
    
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'multi':
        model = SVC(
            C=nu, 
            kernel=kernel, 
            gamma=gamma, 
            class_weight=class_weights,
            probability=True  # Enable probability estimates
        )
        model.fit(X, y)
        
    else:
        # For one-class, only use normal class for training
        normal_idx = y == behaviour
        X_train = X[normal_idx]
        
        # Train model
        model = OneClassSVM(nu = nu, kernel =kernel, gamma = gamma)
        model.fit(X_train)

    return {'model': model, 'scaler': scaler}

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

    print(params_df)
    
    # Convert DataFrame to dictionary format needed for SVM
    best_params = {}
    for _, row in params_df.iterrows():
        behavior = row['behaviour_or_activity']
        best_params[behavior] = {
            'kernel': row['kernel'], 
            'gamma': row['gamma'],
            'C': row['nu']
        }
    
    return best_params

def save_model(model, scaler, file_path):
    """
    Save the trained model and its scaler
    
    Args:
        model: Trained SVM model
        scaler: Fitted StandardScaler
        file_path: Path where to save the model
    """
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save both model and scaler together
    dump({
        'model': model,
        'scaler': scaler
    }, file_path)
    print(f"Model saved to {file_path}")

def train_and_save_SVM(BASE_PATH, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES = None, BEHAVIOUR_SET = None):
    # Load best parameters from CSV
    best_params = load_best_params(Path(f"{BASE_PATH}/Output/Tuning/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_hyperparmaters.csv"))
    
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        for behaviour in TARGET_ACTIVITIES:
            print(f"Training optimal {MODEL_TYPE} {behaviour} SVM model...")
            df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
        
            kernel = best_params[behaviour]['kernel']
            nu = best_params[behaviour]['C']
            gamma = best_params[behaviour]['gamma']

            model_info = train_svm(
                df,
                MODEL_TYPE,
                kernel,
                nu,
                gamma,
                behaviour
            )
            model_path = Path(f"{BASE_PATH}/Output/Models/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
            model = model_info['model']
            scaler = model_info['scaler']
            save_model(model, scaler, model_path)
    else:
        print(f"Training optimal {MODEL_TYPE} {BEHAVIOUR_SET} SVM model...")
        df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}.csv"))
           
        kernel = best_params[BEHAVIOUR_SET]['kernel']
        nu = best_params[BEHAVIOUR_SET]['C']
        gamma = best_params[BEHAVIOUR_SET]['gamma']
           
        model_info = train_svm(
            df,
            MODEL_TYPE,
            kernel,
            nu,
            gamma,
            behaviour = BEHAVIOUR_SET
        )
        model_path = Path(f"{BASE_PATH}/Output/Models/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_model.joblib")
        model = model_info['model']
        scaler = model_info['scaler']
        save_model(model, scaler, model_path)

if __name__ == "__main__":
    train_and_save_SVM(BASE_PATH, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)
