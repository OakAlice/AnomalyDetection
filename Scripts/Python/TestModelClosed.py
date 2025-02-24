from TestModelOpen import predict_single_model
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, roc_auc_score

def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BEHAVIOUR_SET, THRESHOLDING, FOLD):
    print(f"data: {DATASET_NAME}, training: {TRAINING_SET}, model: {MODEL_TYPE}, behaviours: {BEHAVIOUR_SET}, thresholding: {THRESHOLDING}")

    # generate the predictions
    # Determine the predictions file path based on model type and settings
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET.lower() == 'activity':

            predictions_file = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_multi_Activity_{'threshold' if THRESHOLDING else 'NOthreshold'}_closed_predictions.csv")
            model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_multi_Activity_{'threshold' if THRESHOLDING else 'NOthreshold'}_model.joblib")

        else:  # BEHAVIOUR_SET == 'other'
            predictions_file = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_multi_Other_closed_predictions.csv")
            model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_multi_Other_model.joblib")
            # extract the classes present in the model
    else:
        print("this method only works for multiclass models")
    
    saved_data = load(model_path)
    classes = saved_data['model'].classes_

    # only allow those same classes in the test data
    df = pd.read_csv(Path(BASE_PATH) / "Output" / f"fold_{FOLD}" / "Split_data" / f"{DATASET_NAME}_test.csv")
    df = df[df['Activity'].isin(classes)]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    X = df.drop(columns=['Activity', 'ID', 'Time'])
    y = df['Activity']
    metadata = df[['ID', 'Time']]
    
    # then continue everything else the same as for the open models

    if MODEL_TYPE.lower() == 'multi':
        # Generate predictions for multiclass model
        saved_data = load(model_path)
        multiclass_predictions = predict_single_model(X, y, metadata, saved_data['model'], saved_data['scaler'], MODEL_TYPE)
        # account for the threshold in some conditions
        if THRESHOLDING is not False:
            print("accounting for threshold")
            multiclass_predictions['Predicted_Label'] = multiclass_predictions.apply(
                lambda row: row['Predicted_Label'] if row['Best_Probability'] >= 0.5 else "Other", 
                axis=1
            )
 
    # then continue everything else the same as for the open models
    y_true = multiclass_predictions['True_Label']
    y_pred = multiclass_predictions['Predicted_Label']
    labels = sorted(list(set(y_true) | set(y_pred)))
    # labels = sorted(list(set(y_pred))) # to just calculate for the predicted ones
    
    # Get classification report for per-class metrics
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True, labels=labels)  # Remove sample_weight here

    # Calculate weighted metrics manually for the weighted average
    metrics_dict = {}
    for label in labels:
        # Create binary labels for current class
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        count = y_true_binary.sum()
        
        try:
            # Calculate AUC
            auc = roc_auc_score(y_true_binary, y_pred_binary)
        except Exception as e:
            print(f"Warning: Could not calculate AUC for {label}: {str(e)}")
            auc = 0
        
        # Calculate additional metrics
        TP = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        TN = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
        FP = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        FN = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        # Store metrics for this class
        metrics_dict[label] = {
            'AUC': auc,
            'F1': report_dict[label]['f1-score'],
            'Precision': report_dict[label]['precision'],
            'Recall': report_dict[label]['recall'],
            'Support': report_dict[label]['support'],
            'Count': count,
            'Accuracy': accuracy,
            'Specificity': specificity,
            'FPR': fpr
        }
    
    # Calculate true weighted averages based on true class prevalence
    class_frequencies = y_true.value_counts()
    total_samples = class_frequencies.sum()
    valid_labels = [label for label in labels if label in class_frequencies.index]

    metrics_dict['weighted_avg'] = {
        'AUC': sum(metrics_dict[label]['AUC'] * class_frequencies[label] 
                    for label in valid_labels) / total_samples,
        'F1': sum(metrics_dict[label]['F1'] * class_frequencies[label] 
                    for label in valid_labels) / total_samples,
        'Precision': sum(metrics_dict[label]['Precision'] * class_frequencies[label] 
                        for label in valid_labels) / total_samples,
        'Recall': sum(metrics_dict[label]['Recall'] * class_frequencies[label] 
                        for label in valid_labels) / total_samples,
        'Support': total_samples,
        'Count': total_samples,
        'Accuracy': sum(metrics_dict[label]['Accuracy'] * class_frequencies[label]
                        for label in valid_labels) / total_samples,
        'Specificity': sum(metrics_dict[label]['Specificity'] * class_frequencies[label]
                        for label in valid_labels) / total_samples,
        'FPR': sum(metrics_dict[label]['FPR'] * class_frequencies[label]
                        for label in valid_labels) / total_samples
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
            
    # Save metrics
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET == 'Activity':
            if THRESHOLDING is not False:
                metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_closed_metrics.csv")
            else:
                metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_closed_metrics.csv")
        else:
            metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_closed_metrics.csv")
               
    else:
        metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_closed_metrics.csv")
            
    metrics_df.to_csv(metrics_path)            
    return metrics_dict

if __name__ == "__main__":
     main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
