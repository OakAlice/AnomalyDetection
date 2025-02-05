from MainScript import BASE_PATH, THRESHOLDING
from joblib import load
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# test the model
def predict_single_model(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                     behaviour, model, scaler, X, y, ID, Time):
    """
    Make predictions for a single model and save resultsError calculating performance metrics
    
    Args:
        BASE_PATH (str): Base path for saving outputs
        DATASET_NAME (str): Name of the dataset
        TRAINING_SET (str): Training set identifier
        MODEL_TYPE (str): Type of model ('binary', 'oneclass', or 'multi')
        behaviour (str): Behavior name or behavior set name for multiclass
        model: Trained model object
        scaler: Fitted scaler object
        X (DataFrame): Feature data
        y (Series): True labels
        ID (Series): Sample IDs
        Time (Series): Time values
    """
    print(f"\nMaking predictions...")
    
    # Get the feature names used during training from the scaler
    scaler_features = scaler.feature_names_in_
    
    # Ensure X has the same features in the same order as used during training
    missing_features = set(scaler_features) - set(X.columns)
    extra_features = set(X.columns) - set(scaler_features)
    
    if missing_features or extra_features:
        print(f"Warning: Feature mismatch detected")
        if missing_features:
            print(f"Missing features: {missing_features}")
        if extra_features:
            print(f"Extra features: {extra_features}")
        
        # Select only the features used during training, in the same order
        X = X[scaler_features]
    
    # make predictions
    X_scaled = scaler.transform(X)  # Scale test data in the same way
    predictions = model.predict(X_scaled)
    
    # Get probabilities based on model type
    try:
        if MODEL_TYPE.lower() == 'oneclass':
            # For one-class SVM, decision_function gives distance from hyperplane
            # Convert to probability-like scores between 0 and 1 using sigmoid
            decision_scores = model.decision_function(X_scaled)
            probabilities = 1 / (1 + np.exp(-decision_scores))
            # Create DataFrame with single probability column
            prob_df = pd.DataFrame({
                'Probability': probabilities
            })
        else:
            # For binary and multiclass, try predict_proba first
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                classes = model.classes_
                prob_columns = {f'Probability_{cls}': prob for cls, prob in zip(classes, probabilities.T)}
            else:
                # If predict_proba not available, use decision_function
                decision_scores = model.decision_function(X_scaled)
                # Convert to probability-like scores using sigmoid
                if decision_scores.ndim == 1:
                    # Binary case
                    probabilities = 1 / (1 + np.exp(-decision_scores))
                    prob_columns = {
                        f'Probability_{model.classes_[1]}': probabilities,
                        f'Probability_{model.classes_[0]}': 1 - probabilities
                    }
                else:
                    # Multiclass case
                    probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1)[:, np.newaxis]
                    prob_columns = {f'Probability_{cls}': prob for cls, prob in zip(model.classes_, probabilities.T)}
            
            prob_df = pd.DataFrame(prob_columns)
                   
    except Exception as e:
        print(f"Warning: Could not calculate probabilities: {str(e)}")
        # Create dummy probability column with 1.0 for all predictions
        prob_df = pd.DataFrame({'Probability': 1.0}, index=range(len(predictions)))
    
    # Save predictions with probabilities
    results_df = pd.DataFrame({
        'True_Label': y,
        'Predicted_Label': predictions,
        'ID': ID,
        'Time': Time
    })
    
    # Combine predictions with probabilities
    results_df = pd.concat([results_df, prob_df], axis=1)

    results_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv"),
                index=False)
    
    return results_df

# merge the predictions from the individual models
def merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES = None):
    print("\nMerging individual model prediction results...")
    all_predictions = []
            
    for behaviour in TARGET_ACTIVITIES:
        try:
            predictions_path = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_predictions.csv")                 
            predictions_df = pd.read_csv(predictions_path)

            print(predictions_df.head())
            
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

def calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHODLING):
    print("\nCalculating performance metrics...")

    try:
        if THRESHODLING is not False: # any number other than False
            if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
                # modify the predictions based on the confidence
                # need to check whether this still works
                multiclass_predictions = multiclass_predictions[multiclass_predictions['Best_Probability'] >= THRESHODLING]
            else:
                # for multiclass, we just select whether it was above the threshold
                print(list(multiclass_predictions.columns))

                multiclass_predictions['Predicted_Label'] = np.where(
                        # if the highest probability is greater than the threshold, then use that prediction
                        # otherwise, use 'Other'
                        multiclass_predictions['Probability'] >= THRESHODLING,
                        multiclass_predictions['Best_Probability'],
                        'Other'
                )
                 
        # calculate AUC, F1, Precision, Recall, and Accuracy for each behaviour
        y_true = multiclass_predictions['True_Label']
        y_pred = multiclass_predictions['Predicted_Label']
        labels = sorted(list(set(y_true) | set(y_pred)))

        full_cm = confusion_matrix(y_true, y_pred, labels=labels)

        print(full_cm)

        full_labels = sorted(list(set(y_true) | set(y_pred)))

        print(f"model type: {MODEL_TYPE}")
        if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
            # for binary and oneclass, change so everything not in target activities is "Other"
            y_true = y_true.apply(lambda x: x if x in TARGET_ACTIVITIES else 'Other')
            y_pred = y_pred.apply(lambda x: x if x in TARGET_ACTIVITIES else 'Other')
            # Only use target activities and "Other" for labels
            labels = sorted(TARGET_ACTIVITIES + ['Other'])

        # make the plots and save to file
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if MODEL_TYPE.lower() == 'multi':
            if THRESHODLING is not False:
                cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_confusion_matrix.png")
            else:
               cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_confusion_matrix.png") 
        else:
            if THRESHODLING is not False:
                cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_threshold_confusion_matrix.png")
            else:
                cm_path = Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_NOthreshold_confusion_matrix.png")
           
        generate_heatmap_confusion_matrix(cm, labels, TARGET_ACTIVITIES, cm_path)

        # save raw confusion matrix data
        cm_df = pd.DataFrame(full_cm, index = full_labels, columns = full_labels)
        if MODEL_TYPE.lower() == 'multi':
            if THRESHODLING is not False:
                cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_confusion_matrix.csv"))
            else:
               cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_confusion_matrix.csv")) 
        else:
            if THRESHODLING is not False:
                cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_threshold_confusion_matrix.csv"))
            else:
                cm_df.to_csv(Path(f"{BASE_PATH}/Output/Testing/ConfusionMatrices/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_NOthreshold_confusion_matrix.csv"))
            
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

def make_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, behaviour_param=None):
    print(f"behaviour_param is {behaviour_param}")

    """
    Test SVM models with either TARGET_ACTIVITIES or BEHAVIOUR_SETS
    
    Args:
        BASE_PATH: Base path for the project
        DATASET_NAME: Name of the dataset
        TRAINING_SET: Training set identifier
        MODEL_TYPE: Type of model ('binary', 'oneclass', or 'multiclass')
        behaviour_param: Either TARGET_ACTIVITIES list or BEHAVIOUR_SETS list
    """
    print(f"\nStarting SVM testing for {DATASET_NAME} with {MODEL_TYPE} configuration...")
    
    # load in and prepare the test data
    print(f"Loading test data from {DATASET_NAME}_test_features_cleaned.csv")
    file_path = Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_test.csv"
    df = pd.read_csv(file_path)
        
    X = df.drop(columns=['Activity', 'ID'])
    y = df['Activity']
    ID = df['ID']
    Time = df['Time']
    
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        for behaviour in behaviour_param:
            print(f"\nTesting model for {behaviour}...")
            # load in the optimal model
            model_path = Path(f"{BASE_PATH}/Output/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}_model.joblib")
            saved_data = load(model_path)
            model = saved_data['model']
            scaler = saved_data['scaler']

            # test the model, make and save the predictions
            multiclass_predictions = predict_single_model(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                    behaviour, model, scaler, X, y, ID, Time)

        print("\nMerging predictions from all behavior models...")
        multiclass_predictions = merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES=behaviour_param)
    else:
        print(f"Beginning predictions for multiclass {behaviour_param}...")
        model_path = Path(f"{BASE_PATH}/Output/Models/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour_param}_model.joblib")
        saved_data = load(model_path)
        model = saved_data['model']
        scaler = saved_data['scaler']

        multiclass_predictions = predict_single_model(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                    behaviour_param, model, scaler, X, y, ID, Time)
        # Get the actual predicted class with highest probability
        prob_cols = [col for col in multiclass_predictions.columns if 'Probability_' in col]
        multiclass_predictions['Probability'] = multiclass_predictions[prob_cols].max(axis=1)
        multiclass_predictions['Best_Probability'] = multiclass_predictions['Predicted_Label']

    return(multiclass_predictions)


def main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING):
    # BEHAVIOUR_SETS = ["Activity", "Other"]

    try:
        if MODEL_TYPE.lower() == 'multi':
            predictions_file = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_predictions.csv")
            print(f"predictions_file is {predictions_file}")

            if not predictions_file.exists():
                print(f"Beginning preictions for Multiclass")
                multiclass_predictions = make_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, behaviour_param = BEHAVIOUR_SET)
                # save it
                multiclass_predictions.to_csv(predictions_file)

            print("\nCalculating final performance metrics.")
            multiclass_predictions = pd.read_csv(predictions_file)
            metrics_dict = calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
            
        else: # for the binary and oneclass models
            predictions_file = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_all_predictions_merged.csv")
            print(f"predictions_file is {predictions_file}")
            
            if not predictions_file.exists():
                print("No existing predictions found")
                individual_predictions_file = Path(f"{BASE_PATH}/Output/Testing/Predictions/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{TARGET_ACTIVITIES[1]}_predictions.csv")
                print(individual_predictions_file)
                
                if individual_predictions_file.exists():
                    print('merging predictions again')
                    multiclass_predictions = merge_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)
                else:
                    print("Running full model testing...")
                    multiclass_predictions = make_predictions(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, behaviour_param = TARGET_ACTIVITIES)
            
            else:
                multiclass_predictions = pd.read_csv(predictions_file)
            
            metrics_dict = calculate_performance(multiclass_predictions, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
            
        # load in the predictions
        if metrics_dict:  # Check if metrics_dict is not None
            print("\nSaving final metrics...")
            # Convert to DataFrame
            metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
            
            # Save metrics
            if MODEL_TYPE.lower() == 'multi':
                if THRESHOLDING is not False:
                    metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_threshold_metrics.csv")
                else:
                    metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{BEHAVIOUR_SET}_NOthreshold_metrics.csv")
            else:
                if THRESHOLDING is not False:
                    metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_threshold_metrics.csv")
                else:
                    metrics_path = Path(f"{BASE_PATH}/Output/Testing/Metrics/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_NOthreshold_metrics.csv")
                
            
            metrics_df.to_csv(metrics_path)
            print(f"Results saved to {metrics_path}")
        else:
            print("No metrics were calculated. Check the errors above.")
            
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")

if __name__ == "__main__":
    main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)