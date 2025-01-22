from MainScript import BASE_PATH, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
from SVMTrainModel import train_svm

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
    
    try:
        for behaviour in TARGET_ACTIVITIES:
            data_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv")
            df = pd.read_csv(data_path)
                
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



            print(f"\n{'='*50}")
            print(f"Processing {MODEL_TYPE} SVM for {behaviour}...")
            
            try:
                # Load data
                data_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv")
                print(f"Loading data from: {data_path}")
                df = pd.read_csv(data_path)
                print(f"Data loaded successfully. Shape: {df.shape}")
                
                # Prepare features
                X = df.drop(columns=['Activity', 'ID'])
                y = df['Activity']
                
                print("Scaling features...")
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
                if MODEL_TYPE.lower() == 'oneclass':
                    print("Processing one-class SVM...")
                    normal_idx = y == behaviour
                    X_train = X[normal_idx]
                    X_test = X
                    y_test = y
                    print(f"Training data shape: {X_train.shape}")
                    
                    model = OneClassSVM(**best_params[behaviour])
                    print("Training model...")
                    model.fit(X_train)
                    
                    y_pred = model.predict(X_test)
                    y_pred = np.where(y_pred == 1, behaviour, "Other")
                    
                else:
                    print("Processing binary/multiclass SVM...")
                    groups = df['ID']
                    unique_groups = groups.unique()
                    train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)
                    
                    train_idx = groups.isin(train_groups)
                    test_idx = groups.isin(test_groups)
                    
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
                    
                    model = SVC(**best_params[behaviour])
                    print("Training model...")
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                
                print("\nEvaluation Results:")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                conf_matrix = confusion_matrix(y_test, y_pred)
                print(conf_matrix)
                
                # Store results and save predictions
                try:
                    results[behaviour] = {
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': conf_matrix,
                        'model': model,
                        'scaler': scaler
                    }
                    
                    results_df = pd.DataFrame({
                        'True_Label': y_test,
                        'Predicted_Label': y_pred
                    })
                    
                    pred_path = Path(f"{BASE_PATH}/Output/Testing/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv")
                    results_df.to_csv(pred_path, index=False)
                    print(f"Predictions saved to: {pred_path}")
                    
                except Exception as e:
                    print(f"Error saving results for {behaviour}: {str(e)}")
                
            except Exception as e:
                print(f"Error processing behavior {behaviour}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error in train_test_svm: {str(e)}")
        raise
        
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
    try:
        print("Starting SVM training and testing process...")
        
        # Load best parameters
        params_path = Path(f"{BASE_PATH}/Output/Tuning/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_hyperparameters.csv")
        print(f"Loading parameters from: {params_path}")
        best_params = load_best_params(params_path)
        print("Parameters loaded successfully:")
        print(best_params)

        # Train and test binary SVM
        print("\nStarting SVM training...")
        binary_results = train_test_svm(
            DATASET_NAME=DATASET_NAME,
            TRAINING_SET=TRAINING_SET,
            MODEL_TYPE=MODEL_TYPE,
            BASE_PATH=BASE_PATH,
            TARGET_ACTIVITIES=TARGET_ACTIVITIES,
            best_params=best_params
        )
        
        print("\nSaving final results...")
        save_results_to_csv(binary_results, DATASET_NAME, TRAINING_SET, 'binary', BASE_PATH)
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Fatal error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()

