# Train the optimal model

from MainScript import BASE_PATH, ML_METHOD, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import numpy as np
from pathlib import Path

def train_svm(df, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BASE_PATH, TARGET_ACTIVITIES, best_params, behaviour):
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
    print(f"\nTraining {MODEL_TYPE} SVM for {behaviour}...")
    
    # Prepare features
    X = df.drop(columns=['Activity', 'ID'])
    y = df['Activity']
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if MODEL_TYPE.lower() == 'binary':
        # Add class_weight parameter to handle imbalance
        n_other = sum(y == 'Other')
        n_target = sum(y == behaviour)
        class_weights = {
            behaviour: n_other / len(y),
            'Other': n_target / len(y)
        }
        
        # Update parameters with class weights
        params = best_params[behaviour].copy()
        params['class_weight'] = class_weights
        
        model = SVC(**params)
        model.fit(X, y)
        
    elif MODEL_TYPE.lower() == 'oneclass':
        # For one-class, only use normal class for training
        normal_idx = y == behaviour
        X_train = X[normal_idx]
        
        # Train model
        model = OneClassSVM(**best_params[behaviour])
        model.fit(X_train)
        
    else:  # multiclass
        model = SVC(**best_params[behaviour])
        model.fit(X, y)

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

def train_and_save_SVM(BASE_PATH, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE):
    # Load best parameters from CSV
    best_params = load_best_params(Path(f"{BASE_PATH}/Output/Tuning/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_hyperparmaters.csv"))
    
    for behaviour in TARGET_ACTIVITIES:
        # Train optimal binary and multiclass models - including saving to 'Models' folder
        print(f"Training optimal {MODEL_TYPE} {behaviour} SVM model...")
        # Load data
        df = pd.read_csv(Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
    
        model_info = train_svm(
            df,
            DATASET_NAME,
            TRAINING_SET,
            MODEL_TYPE,
            BASE_PATH,
            TARGET_ACTIVITIES,
            best_params,
            behaviour
        )
        model = model_info['model']
        scaler = model_info['scaler']
        model_path = Path(f"{BASE_PATH}/Output/Models/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
        save_model(model, scaler, model_path)

if __name__ == "__main__":
    train_and_save_SVM(BASE_PATH, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE)
