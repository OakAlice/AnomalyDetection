from MainScript import BASE_PATH, ML_METHOD, MODEL_TYPE, TRAINING_SET, TARGET_ACTIVITIES, DATASET_NAME, BEHAVIOUR_SET
from joblib import load
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# test the model
def make_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                 behaviour, model, scaler, X, y, ID, Time):

    # make predictions
    X_scaled = scaler.transform(X)  # Scale test data in the same way
    predictions = model.predict(X_scaled)

    # calculate performance
    y_true = y
    y_pred = predictions
                   
    # Save predictions
    results_df = pd.DataFrame({
        'True_Label': y,
        'Predicted_Label': predictions,
        'ID': ID,
        'Time': Time
    })

    results_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv"),
                index=False)
    
    return results_df

# merge the predictions from the individual models
def merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES = None, BEHAVIOUR_SET = None):
    print("\nMerging individual model prediction results...")
    all_predictions = []
            
    for behaviour in TARGET_ACTIVITIES:
        try:
            predictions_path = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv")                 
            predictions_df = pd.read_csv(predictions_path)
            # Rename prediction columns to include behaviour name
            predictions_df = predictions_df.rename(columns={
                'True_Label': f'True_Label_{behaviour}',
                'Predicted_Label': f'Predicted_Label_{behaviour}'
            })
            all_predictions.append(predictions_df)
                
        except Exception as e:
            print(f"Error loading predictions for {behaviour}: {str(e)}")
            continue
            
    # Merge all prediction dataframes on ID and Time
    if all_predictions:
        print("\nMerging all prediction files...")
        merged_predictions = all_predictions[0]
        for df in all_predictions[1:]:
            merged_predictions = pd.merge(
                merged_predictions, 
                df,
                on=['ID', 'Time'],
                how='outer'
            )

        # Get all columns that contain 'Predicted_Label_'
        def combine_predictions(row):
            # Get all non-'Other' predictions from the row
            non_other_predictions = [
                pred for pred in row 
                if pred != 'Other'
            ]
            # If we found any non-'Other' predictions, return the first one
            # Otherwise return 'Other'
            # TODO: This isn't a good way to do it, but just filling in the space for now
            return non_other_predictions[0] if non_other_predictions else 'Other'

        # Apply the function across all Predicted_Label columns
        predicted_columns = merged_predictions.filter(like='Predicted_Label_').columns
        merged_predictions['Combined_Predicted_Label'] = merged_predictions[predicted_columns].apply(combine_predictions, axis=1)

        # select the columns we want to keep
        merged_predictions = merged_predictions[['ID', 'Time', 'True_Label_Walking', 'Combined_Predicted_Label']]
        merged_predictions.rename(columns={'True_Label_Walking': 'True_Label', 'Combined_Predicted_Label': 'Predicted_Label'}, inplace=True)

        # Save merged predictions
        merged_path = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_all_predictions_merged.csv")
        merged_predictions.to_csv(merged_path, index=False)

        return merged_predictions
    else:
        print("No prediction files were loaded to merge")

def calculate_performance(multiclass_predictions, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES):
    print("\nCalculating performance metrics...")
    try:
        # calculate AUC, F1, Precision, Recall, and Accuracy for each behaviour
        y_true = multiclass_predictions['True_Label']
        y_pred = multiclass_predictions['Predicted_Label']
        labels = sorted(list(set(y_true) | set(y_pred)))

        full_cm = confusion_matrix(y_true, y_pred, labels=labels)
        full_labels = sorted(list(set(y_true) | set(y_pred)))

        print(f"model type: {MODEL_TYPE}")
        if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
            # for binary and oneclass, change so everything not in target activities is "Other"
            y_true = y_true.apply(lambda x: x if x in TARGET_ACTIVITIES else 'Other')
            y_pred = y_pred.apply(lambda x: x if x in TARGET_ACTIVITIES else 'Other')
            # Only use target activities and "Other" for labels
            labels = sorted(TARGET_ACTIVITIES + ['Other'])

        # Calculate a reduced confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Create custom color mask for confusion matrix
        mask = np.zeros_like(cm, dtype=bool)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                if pred_label == "Other":
                    mask[i,j] = true_label not in TARGET_ACTIVITIES
                else:
                    mask[i,j] = (true_label == pred_label)

        # Create confusion matrix visualization
        plt.figure(figsize=(10, 8))
        
        # Create a masked array for different colormaps
        sns.heatmap(cm, 
                   mask=~mask,  # Inverse mask for red cells
                   cmap='Blues',
                   annot=True,
                   fmt='d',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar=False)
        
        sns.heatmap(cm, 
                   mask=mask,  # Original mask for blue cells
                   cmap='Reds',
                   annot=True,
                   fmt='d',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar=False)

        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix plot
        cm_path = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # save raw confusion matrix data
        cm_df = pd.DataFrame(full_cm, index = full_labels, columns = full_labels)
        cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_confusion_matrix.csv"))

        # Get classification report for per-class metrics
        report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True, labels=labels)

        # Convert probabilities for ROC curve calculation
        metrics_dict = {}
        for label in labels:
            # Create binary labels for current class
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            try:
                # Calculate AUC
                auc = roc_auc_score(y_true_binary, y_pred_binary)
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {label}: {str(e)}")
                auc = 0  # Set to 0 instead of None for failed AUC calculations
            
            # Store metrics for this class
            metrics_dict[label] = {
                'AUC': auc,
                'F1': report_dict[label]['f1-score'],
                'Precision': report_dict[label]['precision'],
                'Recall': report_dict[label]['recall'],
                'Support': report_dict[label]['support']
            }
        
        # Add weighted averages
        valid_aucs = [m['AUC'] for m in metrics_dict.values() if m['AUC'] is not None and m['AUC'] > 0]
        metrics_dict['weighted_avg'] = {
            'AUC': np.mean(valid_aucs) if valid_aucs else 0,
            'F1': report_dict['weighted avg']['f1-score'],
            'Precision': report_dict['weighted avg']['precision'],
            'Recall': report_dict['weighted avg']['recall'],
            'Support': report_dict['weighted avg']['support']
        }
        
        return metrics_dict
    
    except Exception as e:
        print(f"Error calculating performance metrics: {str(e)}")
        # Return a basic metrics dictionary with zeros to prevent NoneType errors
        return {
            'error': {
                'AUC': 0,
                'F1': 0,
                'Precision': 0,
                'Recall': 0,
                'Support': 0
            }
        }

def test_SVM(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES):
    print(f"\nStarting SVM testing for {DATASET_NAME} with {MODEL_TYPE} configuration...")
    
    # load in and prepare the test data
    print(f"Loading test data from {DATASET_NAME}_test_features_cleaned.csv...")
    file_path = Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_test_features_cleaned.csv"
    df = pd.read_csv(file_path)
        
    X = df.drop(columns=['Activity', 'ID'])
    y = df['Activity']
    ID = df['ID']
    Time = df['Time']
    
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        for behaviour in TARGET_ACTIVITIES:
            print(f"\nTesting model for {behaviour}...")
            # load in the optimal model
            model_path = Path(f"{BASE_PATH}/Output/Models/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
            saved_data = load(model_path)
            model = saved_data['model']
            scaler = saved_data['scaler']

            # test the model, make and save the predictions
            multiclass_predictions = make_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                    behaviour, model, scaler, X, y, ID, Time)

        print("\nMerging predictions from all behavior models...")
        multiclass_predictions = merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)
    else:
        print("\nProcessing multiclass model...")
        model_path = Path(f"{BASE_PATH}/Output/Models/{ML_METHOD}/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_model.joblib")
        saved_data = load(model_path)
        model = saved_data['model']
        scaler = saved_data['scaler']

        multiclass_predictions = make_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                    behaviour, model, scaler, X, y, ID, Time)

    print("\nCalculating final performance metrics...")
    metrics_dict = calculate_performance(multiclass_predictions, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)

    return(metrics_dict)

if __name__ == "__main__":
    print("Starting model testing pipeline...")
    try:
        if MODEL_TYPE == 'multiclass':
            metrics_dict = test_SVM(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BEHAVIOUR_SET)
        else:
            predictions_file = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_all_predictions_merged.csv")
            if predictions_file.exists():
                print(f"Loading existing predictions from {predictions_file}")
                multiclass_predictions = pd.read_csv(predictions_file)
                metrics_dict = calculate_performance(multiclass_predictions, ML_METHOD, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)
            else:
                print("No existing predictions found. Running full model testing...")
                metrics_dict = test_SVM(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)
        
        if metrics_dict:  # Check if metrics_dict is not None
            print("\nSaving final metrics...")
            # Convert to DataFrame
            metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
            
            # Save metrics
            metrics_path = Path(f"{BASE_PATH}/Output/Testing/{ML_METHOD}/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_metrics.csv")
            metrics_df.to_csv(metrics_path)
            print(f"Results saved to {metrics_path}")
        else:
            print("No metrics were calculated. Check the errors above.")
            
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")