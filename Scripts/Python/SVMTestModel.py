from MainScript import BASE_PATH, ML_METHOD, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME
from joblib import load
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# test the model
def test_SVM(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES):
    # load in the test data # this has already been cleaned
    file_path = Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_test_features_cleaned.csv"
    df = pd.read_csv(file_path)

    X = df.drop(columns=['Activity', 'ID'])
    y = df['Activity']
    ID = df['ID']
    Time = df['Time']
        
    for behaviour in TARGET_ACTIVITIES:
        print(f"testing performance for {behaviour}")
        # load in the optimal model
        model_path = Path(f"{BASE_PATH}/Output/Models/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
        saved_data = load(model_path)
        model = saved_data['model']
        scaler = saved_data['scaler']

        # make predictions
        X_scaled = scaler.transform(X)  # Scale test data in the same way
        predictions = model.predict(X_scaled)

        # calculate performance and save to file
        y_true = y
        y_pred = predictions
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create rows for the DataFrame
        rows = []
        for label in labels:
            if label in report:
                metrics = report[label]
                rows.append({
                    'ML_Method': ML_METHOD,
                    'Dataset_Name': DATASET_NAME,
                    'Training_Set': TRAINING_SET,
                    'Model_Type': MODEL_TYPE,
                    'Behaviour': behaviour,
                    'Class': label,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1-score'],
                    'Support': metrics['support']
                })
        
        # Add summary metrics from the classification report
        for metric in ['accuracy', 'macro avg', 'weighted avg']:
            if metric in report:
                if metric == 'accuracy':
                    rows.append({
                        'ML_Method': ML_METHOD,
                        'Dataset_Name': DATASET_NAME,
                        'Training_Set': TRAINING_SET,
                        'Model_Type': MODEL_TYPE,
                        'Behaviour': behaviour,
                        'Class': metric,
                        'Precision': report[metric],  # accuracy is a single float
                        'Recall': None,
                        'F1_Score': None,
                        'Support': None
                    })
                else:  # for 'macro avg' and 'weighted avg'
                    rows.append({
                        'ML_Method': ML_METHOD,
                        'Dataset_Name': DATASET_NAME,
                        'Training_Set': TRAINING_SET,
                        'Model_Type': MODEL_TYPE,
                        'Behaviour': behaviour,
                        'Class': metric,
                        'Precision': report[metric]['precision'],
                        'Recall': report[metric]['recall'],
                        'F1_Score': report[metric]['f1-score'],
                        'Support': report[metric]['support']
                    })
        
        # save to CSV
        results_df = pd.DataFrame(rows)
        results_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_results.csv"),
                         index=False)
            
        # Save predictions
        results_df = pd.DataFrame({
            'True_Label': y,
            'Predicted_Label': predictions,
            'ID': ID,
            'Time': Time
        })

        results_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv"),
                        index=False)

if __name__ == "__main__":
    test_SVM(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)
