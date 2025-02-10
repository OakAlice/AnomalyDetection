from MainScript import BASE_PATH, THRESHOLDING
from joblib import load
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# merge the predictions from the individual models
def merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES = None):
    print("\nMerging individual model prediction results...")
    all_predictions = []
            
    for behaviour in TARGET_ACTIVITIES:
        try:
            predictions_path = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv")                 
            predictions_df = pd.read_csv(predictions_path)
            
            if MODEL_TYPE.lower() == 'oneclass':
                # Convert -1/1 to Other/behaviour
                predictions_df['Predicted_Label'] = predictions_df['Predicted_Label'].map({-1: 'Other', 1: behaviour})
            
            # Rename columns to include behaviour name
            predictions_df = predictions_df.rename(columns={
                'True_Label': f'True_Label_{behaviour}',
                'Predicted_Label': f'Predicted_Label_{behaviour}'
            })
            
            # Rename probability columns if they exist
            prob_cols = [col for col in predictions_df.columns if 'Probability' in col]
            for col in prob_cols:
                predictions_df = predictions_df.rename(columns={
                    col: f'{col}_{behaviour}'
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

        # deal with conflicts in the collaboration models
        def get_best_prediction(row):
            best_prob = 0
            best_pred = 'Other'
            
            # Check each behaviour's prediction and probability
            for behaviour in TARGET_ACTIVITIES:
                pred_col = f'Predicted_Label_{behaviour}'
                
                # Get probability based on model type
                if MODEL_TYPE.lower() == 'oneclass':
                    prob_col = f'Probability_{behaviour}'
                    prob = row[prob_col] if prob_col in row else 0
                else:
                    # For binary/multiclass, get probability for the predicted class
                    pred = row[pred_col]
                    prob_col = f'Probability_{pred}_{behaviour}'
                    prob = row[prob_col] if prob_col in row else 0
                
                # If this prediction is not 'Other' and has higher probability
                if row[pred_col] != 'Other' and prob > best_prob:
                    best_prob = prob
                    best_pred = row[pred_col]
            
            return pd.Series({
                'Predicted_Label': best_pred,
                'Best_Probability': best_prob
            })

        # Apply the function across all prediction columns
        prediction_probs = merged_predictions.apply(get_best_prediction, axis=1)
        
        # Add the combined predictions and probabilities to the merged dataframe
        merged_predictions['Predicted_Label'] = prediction_probs['Predicted_Label']
        merged_predictions['Best_Probability'] = prediction_probs['Best_Probability']

        # select the columns we want to keep (now including original predictions)
        true_label_col = [col for col in merged_predictions.columns if 'True_Label_' in col][0]  # Get first True_Label column
        pred_cols = [col for col in merged_predictions.columns if 'Predicted_Label_' in col]
        prob_cols = [col for col in merged_predictions.columns if 'Probability_' in col]
        
        columns_to_keep = ['ID', 'Time', true_label_col, 'Predicted_Label', 'Best_Probability'] + pred_cols + prob_cols
        merged_predictions = merged_predictions[columns_to_keep]
        merged_predictions.rename(columns={true_label_col: 'True_Label'}, inplace=True)

        # Save merged predictions
        merged_path = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_all_predictions_merged.csv")
        merged_predictions.to_csv(merged_path, index=False)

        return merged_predictions
    else:
        print("No prediction files were loaded to merge")

def generate_heatmap_confusion_matrix(cm, labels, TARGET_ACTIVITIES, cm_path):
    # Create custom color mask for confusion matrix
    mask = np.zeros_like(cm, dtype=bool)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if pred_label == "Other":
                mask[i,j] = true_label not in TARGET_ACTIVITIES
            else:
                mask[i,j] = (true_label == pred_label)

    # Create mask for zero values
    zero_mask = (cm == 0)

    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
            
    # Create a masked array for different colormaps
    sns.heatmap(cm, 
            mask=~mask | zero_mask,  # Inverse mask for red cells, plus zero mask
            cmap='Blues',
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False)
            
    sns.heatmap(cm, 
            mask=mask | zero_mask,  # Original mask for blue cells, plus zero mask
            cmap='Reds',
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False)

    # Add white squares for zeros
    sns.heatmap(cm,
            mask=~zero_mask,  # Show only zeros
            cmap=['white'],  # Use white color
            annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False)

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
 
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()

def generate_confusion_matrices(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHODLING):
    y_true = multiclass_predictions['True_Label']
    y_pred = multiclass_predictions['Predicted_Label']
    labels = sorted(list(set(y_true) | set(y_pred)))

    full_cm = confusion_matrix(y_true, y_pred, labels=labels)

    full_labels = sorted(list(set(y_true) | set(y_pred)))

    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        # for binary and oneclass, change so everything not in target activities is "Other"
        y_true = y_true.apply(lambda x: x if x in TARGET_ACTIVITIES else 'Other')
        y_pred = y_pred.apply(lambda x: x if x in TARGET_ACTIVITIES else 'Other')
        # Only use target activities and "Other" for labels
        labels = sorted(TARGET_ACTIVITIES + ['Other'])

    # make the plots and save to file
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET.lower() == 'activity':
            if THRESHODLING is not False:
                cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_confusion_matrix.png")
            else:
                cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_confusion_matrix.png") 
        else:
            cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_confusion_matrix.png")
    else:
        cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_confusion_matrix.png")
        
    generate_heatmap_confusion_matrix(cm, labels, TARGET_ACTIVITIES, cm_path)

    # save raw confusion matrix data
    cm_df = pd.DataFrame(full_cm, index = full_labels, columns = full_labels)
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET.lower() == 'activity':
            if THRESHODLING is not False:
                cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_confusion_matrix.csv"))
            else:
                cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_confusion_matrix.csv")) 
        else:
            cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_confusion_matrix.csv"))
    else:
            cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_confusion_matrix.csv"))
        
def calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHODLING):
    y_true = multiclass_predictions['True_Label']
    y_pred = multiclass_predictions['Predicted_Label']
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Get classification report for per-class metrics
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True, labels=labels)  # Remove sample_weight here

    print(report_dict)

    # Calculate weighted metrics manually for the weighted average
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
            auc = 0
        
        # Store metrics for this class
        metrics_dict[label] = {
            'AUC': auc,
            'F1': report_dict[label]['f1-score'],
            'Precision': report_dict[label]['precision'],
            'Recall': report_dict[label]['recall'],
            'Support': report_dict[label]['support']
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
        'Support': total_samples
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
            
    # Save metrics
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET == 'Activity':
            if THRESHOLDING is not False:
                metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_metrics.csv")
            else:
                metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_metrics.csv")
        else:
            metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_metrics.csv")
               
    else:
        metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_metrics.csv")
            
    metrics_df.to_csv(metrics_path)            
    return metrics_dict

def generate_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING):
    try:

        # Determine the predictions file path based on model type and settings
        if MODEL_TYPE.lower() == 'multi':
            if BEHAVIOUR_SET.lower() == 'activity':

                predictions_file = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_multi_Activity_{'threshold' if THRESHOLDING else 'NOthreshold'}_predictions.csv")
                model_path = Path(f"{BASE_PATH}/Output/Models/{DATASET_NAME}_{TRAINING_SET}_multi_Activity_{'threshold' if THRESHOLDING else 'NOthreshold'}_model.joblib")

            else:  # BEHAVIOUR_SET == 'other'
                predictions_file = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_multi_Other_predictions.csv")
                model_path = Path(f"{BASE_PATH}/Output/Models/{DATASET_NAME}_{TRAINING_SET}_multi_Other_model.joblib")
        else:  # binary or oneclass
            predictions_file = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_all_predictions_merged.csv")

        # If predictions already exist, load them and exit function
        if predictions_file.exists():
            print("predictions already generated")
            multiclass_predictions = pd.read_csv(predictions_file)
            return multiclass_predictions

        # Load test data
        print(f"Loading test data from {DATASET_NAME}_test.csv")
        df = pd.read_csv(Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_test.csv")
        X = df.drop(columns=['Activity', 'ID', 'Time'])
        y = df['Activity']
        metadata = df[['ID', 'Time']]

        if MODEL_TYPE.lower() == 'multi':
            # Generate predictions for multiclass model
            saved_data = load(model_path)
            predictions_df = predict_single_model(X, y, metadata, saved_data['model'], saved_data['scaler'])
            # account for the threshold in some conditions
            if THRESHOLDING is not False:
                print("accounting for threshold")
                predictions_df['Predicted_Label'] = predictions_df.apply(
                    lambda row: row['Predicted_Label'] if row['Best_Probability'] >= 0.5 else "Other", 
                    axis=1
                )
            predictions_df.to_csv(predictions_file, index=False)
            return predictions_df

        else:
            # Generate predictions for each behaviour in binary/oneclass models
            all_predictions = []
            for behaviour in TARGET_ACTIVITIES:
                model_path = Path(f"{BASE_PATH}/Output/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
                saved_data = load(model_path)
                predictions = predict_single_model(X, y, metadata, saved_data['model'], saved_data['scaler'])
                predictions.to_csv(Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv"), index=False)
                all_predictions.append(predictions)

            # Merge predictions from all behaviors
            multiclass_predictions = merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)
            return multiclass_predictions

    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return None
    

def predict_single_model(X, y, metadata, model, scaler):
    """Helper function to generate predictions for a single model"""
    # Scale features
    scaler_features = scaler.feature_names_in_
    scaler_features = [feat for feat in scaler_features if feat != "Time"]  # Remove Time from features

    X = X[scaler_features]  # Ensure columns match
    
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)

    print(f"predictions made {predictions}")
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        prob_columns = {f'Probability_{cls}': prob for cls, prob in zip(model.classes_, probabilities.T)}
    else:
        decision_scores = model.decision_function(X_scaled)
        if decision_scores.ndim == 1:  # Binary/OneClass
            probabilities = 1 / (1 + np.exp(-decision_scores))
            prob_columns = {'Probability': probabilities}
        else:  # Multiclass
            probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1)[:, np.newaxis]
            prob_columns = {f'Probability_{cls}': prob for cls, prob in zip(model.classes_, probabilities.T)}
    
    # Create results dataframe with metadata first
    results = pd.DataFrame()
    for col in ['ID', 'Time']:
        if col in metadata.columns:
            results[col] = metadata[col]
    
    # Add predictions and probabilities
    results['True_Label'] = y
    results['Predicted_Label'] = predictions
    for col, values in prob_columns.items():
        results[col] = values

    # add column for the probability of the predicted label
    results['Best_Probability'] = results.apply(lambda row: row[f"Probability_{row['Predicted_Label']}"], axis=1)
    
    return results

def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING):

    # generate the predictions
    multiclass_predictions = generate_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING) 
    # generate the confusion matrices
    generate_confusion_matrices(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
    # calculate the metrics
    calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)