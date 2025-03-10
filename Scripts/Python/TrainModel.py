# Train the optimal model

from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os

def train_svm(df, MODEL_TYPE, kernel, nu, gamma, behaviour):
    print(f"\nTraining {MODEL_TYPE} SVM for {behaviour}...")

    df = df.dropna()
    # Prepare features after cleaning
    print("removing metadata columns now")
    X = df.drop(columns=['Activity', 'Time', 'ID'])
    y = df['Activity']
    
    # Verify no NaN values in features
    if X.isna().any().any():
        raise ValueError("NaN values still present in features after cleaning")
    
    # Rest of your existing code remains the same
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Add class_weight parameter to handle imbalance
    if MODEL_TYPE.lower() == 'binary':
        n_other = sum(y == 'Other')
        print(f"N-other: {n_other}")
        n_target = sum(y == behaviour)
        print(f"N-target: {n_target}")
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
        print("beginning to fit model")
        model.fit(X, y)
        
    else:
        # For one-class, only use normal class for training
        normal_idx = y == behaviour
        X_train = X[normal_idx]
        
        # Train model
        model = OneClassSVM(
            nu=min(0.99, 1/nu),  # for the oneclass, the nu has to be between 0 -1 so have to convert
            kernel=kernel,
            gamma=gamma
        )
        model.fit(X_train)

    return {'model': model, 'scaler': scaler}

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


def find_matching_file(path_pattern):
    # Convert Path object to string for glob
    path_pattern = str(path_pattern)
    
    # Find the file case-insensitively
    matching_files = glob.glob(path_pattern, recursive=True)

    # If no exact match found, try case-insensitive search
    if not matching_files:
        directory = os.path.dirname(path_pattern)
        filename = os.path.basename(path_pattern)
        for file in os.listdir(directory):
            if file.lower() == filename.lower():
                matching_files = [os.path.join(directory, file)]
                break

    # Read the file
    if matching_files:
        print(matching_files)
        df = pd.read_csv(matching_files[0])  # Added [0] to get the first match
    else:
        raise FileNotFoundError(f"No matching file found for pattern: {path_pattern}")

    return df

def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES = None, BEHAVIOUR_SET = None, THRESHOLDING = False, FOLD = 0):
    try:
        # load the hyperparameters
        parameters_path = f"{BASE_PATH}/Output/fold_{FOLD}/Tuning/Combined_optimisation_results.csv"
        params = pd.read_csv(parameters_path)

        print(f"parameters_path: {parameters_path}")

        print(f"params: {params.tail()}")
        
        # Add debug prints to check the filtering conditions
        print("\nFiltering conditions:")
        print(f"Dataset name: {DATASET_NAME}")
        print(f"Training set: {TRAINING_SET}")
        print(f"Model type: {MODEL_TYPE}")
        print(f"Thresholding: {THRESHOLDING}")
        print(f"Behaviour set: {BEHAVIOUR_SET}")

        relevant_params = params[
            (params['dataset_name'] == DATASET_NAME) & 
            (params['training_set'] == TRAINING_SET) & 
            (params['model_type'] == MODEL_TYPE) &
            (params['thresholding'] == THRESHOLDING)
        ]
        
        # Add debug print to check filtered results
        print(f"\nNumber of matching parameter rows: {len(relevant_params)}")
        if len(relevant_params) == 0:
            raise ValueError("No matching parameters found in the optimization results file")

        if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
            for behaviour in TARGET_ACTIVITIES:
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
                
                if model_path.exists():
                    print(f"Model {model_path} already exists, skipping")
                    continue

                df = find_matching_file(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv"))
                print(f"unique activities: {df['Activity'].unique()}")
                print(f"unique ids: {df['ID'].unique()}")
                # Take up to 200 samples per group, or all available if fewer

                df = df.groupby(['ID', 'Activity']).apply(
                    lambda x: x.sample(n=min(len(x), 1000), replace=False)
                ).reset_index(drop=True)

                behaviour_params = relevant_params[relevant_params['behaviour'] == behaviour]
                kernel = behaviour_params['kernel'].iloc[0]
                nu = float(behaviour_params['C'].iloc[0])
                gamma = float(behaviour_params['gamma'].iloc[0])

                print(behaviour_params)

                model_info = train_svm(
                    df,
                    MODEL_TYPE,
                    kernel,
                    nu,
                    gamma,
                    behaviour
                )

                model = model_info['model']
                scaler = model_info['scaler']
                save_model(model, scaler, model_path)
        else:
            print(f"Behaviour set: {BEHAVIOUR_SET}")
            if BEHAVIOUR_SET.lower() == 'activity' and THRESHOLDING is not False:
                threshold = 'threshold'
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_{threshold}_model.joblib")
            elif BEHAVIOUR_SET.lower() == 'activity' and THRESHOLDING is False:
                threshold = 'NOthreshold'
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_{threshold}_model.joblib")
            else:
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_Other_model.joblib")
            
            if model_path.exists():
                print(f"Model {model_path} already exists, skipping")
                return

            df = find_matching_file(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}.csv"))
            print(f"path: {Path(f"{BASE_PATH}/Output/fold_{FOLD}/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}.csv")}")

            df = df.groupby(['ID', 'Activity']).apply(
                        lambda x: x.sample(n=min(len(x), 1000), replace=False)
                    ).reset_index(drop=True)
            
            model_params = relevant_params[relevant_params['behaviour'] == BEHAVIOUR_SET]

            print(f"\nParameters for behaviour set {BEHAVIOUR_SET}:")
            print(model_params)
            
            if len(model_params) == 0:
                print(f"\nAvailable behaviours in filtered params:")
                print(relevant_params['behaviour'].unique())
                raise ValueError(f"No parameters found for behaviour set: {BEHAVIOUR_SET}")

            kernel = model_params['kernel'].iloc[0]
            nu = float(model_params['C'].iloc[0])
            gamma = float(model_params['gamma'].iloc[0])
            
            model_info = train_svm(
                df,
                MODEL_TYPE,
                kernel,
                nu,
                gamma,
                behaviour = BEHAVIOUR_SET
            )
            
            model = model_info['model']
            scaler = model_info['scaler']
            save_model(model, scaler, model_path)
    
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)