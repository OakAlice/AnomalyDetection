from MainScript import BASE_PATH, THRESHOLDING
from joblib import load
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# merge the predictions from the individual models
def merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES=None, FOLD=None):
    print("\nMerging individual model prediction results...")
    all_predictions = []
            
    for behaviour in TARGET_ACTIVITIES:
        try:
            predictions_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv")                 
            predictions_df = pd.read_csv(predictions_path)
            
            if MODEL_TYPE.lower() == 'oneclass':
                # Rename columns to include the behaviour name for one-class models
                predictions_df = predictions_df.rename(columns={
                    'Predicted_Label': f'Predicted_{behaviour}',
                    'Probability': f'Probability_{behaviour}'
                })
            elif MODEL_TYPE.lower() == 'binary':
                # For binary models, select only the desired columns and rename the predicted label
                predictions_df = predictions_df[['ID', 'Time', 'True_Label', 'Predicted_Label', f'Probability_{behaviour}']]
                predictions_df = predictions_df.rename(columns={'Predicted_Label': f'Predicted_{behaviour}'})
        
            all_predictions.append(predictions_df)
                
        except Exception as e:
            print(f"Error loading predictions for {behaviour}: {str(e)}")
            continue
            
    if not all_predictions:
        print("No prediction files were loaded to merge")
        return None

    # Start with a copy of the first predictions dataframe (using its True_Label)
    merged_predictions = all_predictions[0].copy()
    # For every subsequent prediction dataframe, drop the True_Label column to avoid duplication
    for df in all_predictions[1:]:
        if 'True_Label' in df.columns:
            df = df.drop(columns=['True_Label'])
        merged_predictions = pd.merge(
            merged_predictions, 
            df,
            on=['ID', 'Time'],
            how='outer'
        )

    # Updated helper function: iterate through each target behaviour and choose the one with the highest probability.
    def get_best_prediction(row):
        best_prob = 0
        best_pred = 'Other'
        
        # Loop through each behaviour and update if a higher probability is found.
        for behaviour in TARGET_ACTIVITIES:
            prob_col = f'Probability_{behaviour}'
            pred_col = f'Predicted_{behaviour}'
            prob = row[prob_col] if (prob_col in row and not pd.isnull(row[prob_col])) else 0
            if prob > best_prob:
                best_prob = prob
                best_pred = row[pred_col] if pred_col in row else 'Other'
        return pd.Series({
            'Predicted_Label': best_pred,
            'Best_Probability': best_prob
        })

    # Apply the function to determine the final prediction and its probability.
    prediction_probs = merged_predictions.apply(get_best_prediction, axis=1)
    merged_predictions['Predicted_Label'] = prediction_probs['Predicted_Label']
    merged_predictions['Best_Probability'] = prediction_probs['Best_Probability']

    # Ensure that we have the true label column. If a column named "True_Label" exists, use it;
    # otherwise fall back to the first column that contains "True_Label" in its name.
    if 'True_Label' in merged_predictions.columns:
        true_label_col = 'True_Label'
    else:
        true_label_candidates = [col for col in merged_predictions.columns if 'True_Label_' in col]
        true_label_col = true_label_candidates[0] if true_label_candidates else 'True_Label'

    # Gather individual prediction and probability columns for reference.
    pred_cols = [col for col in merged_predictions.columns if col.startswith('Predicted_') and col != 'Predicted_Label']
    prob_cols = [col for col in merged_predictions.columns if col.startswith('Probability_')]
    
    columns_to_keep = ['ID', 'Time', true_label_col, 'Predicted_Label', 'Best_Probability'] + pred_cols + prob_cols
    merged_predictions = merged_predictions[columns_to_keep]
    if true_label_col != 'True_Label':
        merged_predictions.rename(columns={true_label_col: 'True_Label'}, inplace=True)

    return merged_predictions

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

def generate_confusion_matrices(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHODLING, REASSIGN_LABELS, FOLD):
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
                cm_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_confusion_matrix.png")
            else:
                if REASSIGN_LABELS is not False:
                    cm_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_confusion_matrix.png") 
                else:
                    cm_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_confusion_matrix.png") 
        else:
            cm_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_confusion_matrix.png")
    else:
        cm_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_confusion_matrix.png")
        
    generate_heatmap_confusion_matrix(cm, labels, TARGET_ACTIVITIES, cm_path)

    # save raw confusion matrix data
    cm_df = pd.DataFrame(full_cm, index = full_labels, columns = full_labels)
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET.lower() == 'activity':
            if THRESHODLING is not False:
                cm_df.to_csv(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_confusion_matrix.csv"))
            else:
                if REASSIGN_LABELS is not False:
                    cm_df.to_csv(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_confusion_matrix.csv")) 
                else:
                    cm_df.to_csv(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_confusion_matrix.csv")) 
        else:
            cm_df.to_csv(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_confusion_matrix.csv"))
    else:
            cm_df.to_csv(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_confusion_matrix.csv"))
        
def calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS, FOLD):
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
        
        # Store metrics for this class
        metrics_dict[label] = {
            'AUC': auc,
            'F1': report_dict[label]['f1-score'],
            'Precision': report_dict[label]['precision'],
            'Recall': report_dict[label]['recall'],
            'Support': report_dict[label]['support'],
            'Count': count
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
        'Count': total_samples
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
            
    # Save metrics
    if MODEL_TYPE.lower() == 'multi':
        if BEHAVIOUR_SET == 'Activity':
            if THRESHOLDING is not False:
                metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_metrics.csv")
            else:
                if REASSIGN_LABELS is not False:    
                    metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_metrics.csv")
                else:
                    metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_fullclasses_metrics.csv")
        else:
            metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_metrics.csv")
               
    else:
        metrics_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_metrics.csv")
            
    metrics_df.to_csv(metrics_path)            
    return metrics_dict

def generate_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD):
    try:

        # Determine the predictions file path based on model type and settings
        if MODEL_TYPE.lower() == 'multi':
            if BEHAVIOUR_SET.lower() == 'activity':

                predictions_file = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_multi_Activity_{'threshold' if THRESHOLDING else 'NOthreshold'}_predictions.csv")
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_multi_Activity_{'threshold' if THRESHOLDING else 'NOthreshold'}_model.joblib")

            else:  # BEHAVIOUR_SET == 'other'
                predictions_file = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_multi_Other_predictions.csv")
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_multi_Other_model.joblib")
            
            if model_path.exists() is False:
                print(f"Model path {model_path} does not exist. Skipping this test")
                return
        
        else:  # binary or oneclass
            predictions_file = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_all_predictions_merged.csv")

        # If predictions already exist, load them and exit function
        if predictions_file.exists():
            print("predictions already generated")
            multiclass_predictions = pd.read_csv(predictions_file)
            return multiclass_predictions

        # Load test data
        print(f"Loading test data from {DATASET_NAME}_test.csv")
        df = pd.read_csv(Path(BASE_PATH) / "Output" / f"fold_{FOLD}" / "Split_data" / f"{DATASET_NAME}_test.csv")

        # print the nsmaes of  the columns wheich contain NaN values # remove these columns
        # print(df.columns[df.isna().any()])
        df = df.dropna(axis=1)

        X = df.drop(columns=['Activity', 'ID', 'Time'])
        print(X.head())
        y = df['Activity']
        print(y.head())
        metadata = df[['ID', 'Time']]
    
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

        else:
            # Generate predictions for each behaviour in binary/oneclass models
            all_predictions = []
            for behaviour in TARGET_ACTIVITIES:
                model_path = Path(f"{BASE_PATH}/Output/fold_{FOLD}/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
                
                if model_path.exists() is False:
                    print(f"Model path {model_path} does not exist. Skipping this test")
                    return
                
                saved_data = load(model_path)
                predictions = predict_single_model(X, y, metadata, saved_data['model'], saved_data['scaler'], MODEL_TYPE, behaviour)
                predictions.to_csv(Path(f"{BASE_PATH}/Output/fold_{FOLD}/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv"), index=False)
                all_predictions.append(predictions)

            # Merge predictions from all behaviors
            multiclass_predictions = merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, FOLD)
        
        # otherwise, keep the original y for the baseline control case  
        multiclass_predictions.to_csv(predictions_file, index=False)
        return multiclass_predictions

    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        return None
    

def predict_single_model(X, y, metadata, model, scaler, MODEL_TYPE, target_class=None):
    """Helper function to generate predictions for a single model"""
    # Scale features
    scaler_features = scaler.feature_names_in_
    scaler_features = [feat for feat in scaler_features if feat != "Time"]  # Remove Time from features

    X = X[scaler_features]  # Ensure columns match
    
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # rename the values for the one-class scenario
    if MODEL_TYPE.lower() == 'oneclass':
        predictions = np.where(predictions == -1, 'Other', target_class)

    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_scaled)
        prob_columns = {f'Probability_{cls}': prob for cls, prob in zip(model.classes_, probabilities.T)}
    else:
        # When using decision_function (common for one-class scenarios)
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

    # add column for the probability of the predicted label in multi case
    if MODEL_TYPE.lower() == 'multi':
        results['Best_Probability'] = results.apply(lambda row: row[f"Probability_{row['Predicted_Label']}"], axis=1)

    return results

def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS, FOLD):
    print(f"Generating results for {TRAINING_SET} with {MODEL_TYPE} model")
    try:
        # generate the predictions
        multiclass_predictions = generate_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD) 
        
        if REASSIGN_LABELS is not False:
            print("reassigning labels")
        # reassign the labels to add in the "Other class"
            if MODEL_TYPE.lower() != 'multi':
                # for one-class and binary class, everything not in target activities is "Other"
                multiclass_predictions['True_Label'] = multiclass_predictions['True_Label'].apply(lambda x: 'Other' if x not in TARGET_ACTIVITIES else x)
                multiclass_predictions['Predicted_Label'] = multiclass_predictions['Predicted_Label'].apply(lambda x: 'Other' if x not in TARGET_ACTIVITIES else x)
            else:
                # for multi class, everything not in trained behaviours is "Other"
                possible_behaviours = list(set(multiclass_predictions['Predicted_Label']))
                print(f"possible behaviours: {possible_behaviours}")
                multiclass_predictions['True_Label'] = multiclass_predictions['True_Label'].apply(lambda x: 'Other' if x not in possible_behaviours else x)
                multiclass_predictions['Predicted_Label'] = multiclass_predictions['Predicted_Label'].apply(lambda x: 'Other' if x not in possible_behaviours else x)
        else:
            # do nothing
            print("not reassigning labels")
            print(multiclass_predictions['True_Label'].unique())
        
        # generate the confusion matrices
        generate_confusion_matrices(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS, FOLD)
        # calculate the metrics
        calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS, FOLD)

    except Exception as e:
        print(f"Error: {e}. Skipping iteration")
        return

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)