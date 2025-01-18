import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List, Dict, Union, Tuple
import random


def calculate_zero_rate_baseline(ground_truth_labels: List[str], model_type: str, target_class: str = None) -> Dict[str, float]:
    """Calculate zero rate baseline metrics for dichotomous models."""
    if model_type in ["OCC", "Binary"]:
        # For OCC and Binary, always predict the non-target class
        zero_rate_preds = ["Other"] * len(ground_truth_labels)
        
        # Calculate metrics
        f1 = f1_score(ground_truth_labels, zero_rate_preds, pos_label=target_class, average='binary')
        precision = precision_score(ground_truth_labels, zero_rate_preds, pos_label=target_class, average='binary')
        recall = recall_score(ground_truth_labels, zero_rate_preds, pos_label=target_class, average='binary')
        accuracy = accuracy_score(ground_truth_labels, zero_rate_preds)
        
        return {
            'F1_Score': f1,
            'Precision': precision, 
            'Recall': recall,
            'Accuracy': accuracy
        }
    else:
        print("This doesn't work for multiclass, use other one")
        return {}


def multiclass_class_metrics(ground_truth_labels: List[str], predictions: List[str]) -> Dict:
    """Calculate performance metrics for multiclass classification."""
    # Get unique classes
    unique_classes = sorted(list(set(predictions) | set(ground_truth_labels)))
    
    # Calculate class prevalence
    prevalence = pd.Series(ground_truth_labels).value_counts(normalize=True)
    prevalence_df = pd.DataFrame({'Class': prevalence.index, 'Prevalence': prevalence.values})
    
    # Calculate per-class metrics
    class_metrics = []
    for cls in unique_classes:
        binary_true = [1 if x == cls else 0 for x in ground_truth_labels]
        binary_pred = [1 if x == cls else 0 for x in predictions]
        
        metrics = {
            'Class': cls,
            'F1_Score': f1_score(binary_true, binary_pred, average='binary', zero_division=0),
            'Precision': precision_score(binary_true, binary_pred, average='binary', zero_division=0),
            'Recall': recall_score(binary_true, binary_pred, average='binary', zero_division=0),
            'Accuracy': accuracy_score(binary_true, binary_pred)
        }
        class_metrics.append(metrics)
    
    class_metrics_df = pd.DataFrame(class_metrics)
    class_metrics_df = pd.merge(class_metrics_df, prevalence_df, on='Class', how='left')
    
    # Replace NaN with 0
    class_metrics_df = class_metrics_df.fillna(0)
    
    # Calculate weighted averages
    weighted_metrics = {
        metric: np.sum(class_metrics_df[metric] * class_metrics_df['Prevalence'])
        for metric in ['F1_Score', 'Precision', 'Recall', 'Accuracy']
    }
    
    # Calculate macro averages
    macro_metrics = {
        metric: np.mean(class_metrics_df[metric])
        for metric in ['F1_Score', 'Precision', 'Recall', 'Accuracy']
    }
    
    return {
        'macro_metrics': macro_metrics,
        'weighted_metrics': weighted_metrics,
        'class_metrics': class_metrics_df
    }


def prepare_test_data(test_feature_data: pd.DataFrame, selected_features: List[str], behaviour: str = None) -> Dict:
    """Prepare test data for model evaluation."""
    # Select features and metadata columns
    all_cols = selected_features + ['Activity', 'Time', 'ID']
    selected_data = test_feature_data[all_cols].copy()
    selected_data = selected_data.dropna()
    
    # Extract labels and metadata
    ground_truth_labels = selected_data['Activity'].values
    time_values = selected_data['Time'].values
    ID_values = selected_data['ID'].values
    
    # Get numeric features only
    numeric_data = selected_data.drop(['Activity', 'Time', 'ID'], axis=1)
    
    # Remove invalid rows
    valid_mask = numeric_data.notna().all(axis=1) & np.isfinite(numeric_data).all(axis=1)
    numeric_data = numeric_data[valid_mask]
    ground_truth_labels = ground_truth_labels[valid_mask]
    time_values = time_values[valid_mask]
    ID_values = ID_values[valid_mask]
    
    return {
        'numeric_data': numeric_data,
        'ground_truth_labels': ground_truth_labels,
        'time_values': time_values,
        'ID_values': ID_values
    }


def save_results(results: pd.DataFrame, predictions: List[str], ground_truth_labels: List[str], 
                time_values: List, ID_values: List, dataset_name: str, model_name: str, base_path: str):
    """Save model results and predictions."""
    import os
    
    # Save performance metrics
    results.to_csv(os.path.join(base_path, "Output", "Testing", 
                               f"{dataset_name}_{model_name}_test_performance.csv"))
    
    # Save predictions
    output = pd.DataFrame({
        'Time': time_values,
        'ID': ID_values,
        'Ground_truth': ground_truth_labels,
        'Predictions': predictions
    })
    
    if len(output) > 0:
        output.to_csv(os.path.join(base_path, "Output", "Testing", "Predictions",
                                 f"{dataset_name}_{model_name}_predictions.csv"))


def random_baseline_metrics(ground_truth_labels: List[str], iterations: int = 100, model: str = "Multi") -> Dict:
    """Calculate random baseline metrics."""
    # Get class information
    classes = sorted(list(set(ground_truth_labels)))
    class_props = pd.Series(ground_truth_labels).value_counts(normalize=True)
    n_classes = len(classes)
    n_samples = len(ground_truth_labels)
    
    class_prop_equal = [1/n_classes] * n_classes
    
    class_metrics_prev = []
    class_metrics_equal = []
    
    for _ in range(iterations):
        # Generate predictions with true class proportions
        random_preds_prev = np.random.choice(classes, size=n_samples, p=class_props)
        random_preds_equal = np.random.choice(classes, size=n_samples, p=class_prop_equal)
        
        metrics_prev = multiclass_class_metrics(ground_truth_labels, random_preds_prev)
        metrics_equal = multiclass_class_metrics(ground_truth_labels, random_preds_equal)
        
        class_metrics_prev.append(metrics_prev['class_metrics'])
        class_metrics_equal.append(metrics_equal['class_metrics'])
    
    # Calculate averages
    class_metrics_combined_prev = pd.concat(class_metrics_prev)
    class_metrics_combined_equal = pd.concat(class_metrics_equal)
    
    averages_prev = class_metrics_combined_prev.groupby('Class').mean()
    averages_equal = class_metrics_combined_equal.groupby('Class').mean()
    
    if model in ["OCC", "Binary"]:
        selected_metrics_prev = averages_prev[averages_prev.index != "Other"]
        selected_metrics_equal = averages_equal[averages_equal.index != "Other"]
    else:
        # Calculate weighted averages
        selected_metrics_prev = (averages_prev * averages_prev['Prevalence'].values[:, None]).sum()
        selected_metrics_equal = (averages_equal * averages_equal['Prevalence'].values[:, None]).sum()
    
    return {
        'macro_summary': {
            'F1_Score_prev': selected_metrics_prev['F1_Score'],
            'Precision_prev': selected_metrics_prev['Precision'],
            'Recall_prev': selected_metrics_prev['Recall'],
            'Accuracy_prev': selected_metrics_prev['Accuracy'],
            'F1_Score_equal': selected_metrics_equal['F1_Score'],
            'Precision_equal': selected_metrics_equal['Precision'],
            'Recall_equal': selected_metrics_equal['Recall'],
            'Accuracy_equal': selected_metrics_equal['Accuracy']
        },
        'class_summary_prev': averages_prev.reset_index(),
        'class_summary_equal': averages_equal.reset_index()
    }


def calculate_full_multi_performance(ground_truth_labels: List[str], predictions: List[str], model: str) -> pd.DataFrame:
    """Calculate comprehensive performance metrics including baselines."""
    # 1. Absolute scores
    macro_scores = multiclass_class_metrics(ground_truth_labels, predictions)
    class_metrics_df = macro_scores['class_metrics']
    weighted_metrics = macro_scores['weighted_metrics']
    
    # 2. Zero Rate baseline
    majority_class = pd.Series(ground_truth_labels).mode()[0]
    zero_rate_preds = [majority_class] * len(ground_truth_labels)
    zero_rate_baseline = multiclass_class_metrics(ground_truth_labels, zero_rate_preds)
    
    # 3. Random baseline
    random_multiclass = random_baseline_metrics(ground_truth_labels, iterations=100, model=model)
    
    # Calculate false positives
    confusion = pd.crosstab(pd.Series(predictions), pd.Series(ground_truth_labels))
    false_positives = {cls: confusion.loc[cls].sum() - confusion.loc[cls, cls] 
                      for cls in confusion.index}
    
    # Compile results
    results = []
    
    # Add macro average results
    macro_row = {
        'Dataset': dataset_name,
        'Model': model,
        'Activity': 'WeightedMacroAverage',
        'Prevalence': None,
        'FalsePositives': None
    }
    macro_row.update(weighted_metrics)
    macro_row.update({f'ZeroR_{k}': v for k, v in zero_rate_baseline['macro_metrics'].items()})
    macro_row.update(random_multiclass['macro_summary'])
    results.append(macro_row)
    
    # Add per-class results
    for activity in set(ground_truth_labels):
        class_row = {
            'Dataset': dataset_name,
            'Model': model,
            'Activity': activity,
            'Prevalence': class_metrics_df.loc[class_metrics_df['Class'] == activity, 'Prevalence'].iloc[0],
            'FalsePositives': false_positives.get(activity, 0)
        }
        
        # Add metrics
        metrics = class_metrics_df[class_metrics_df['Class'] == activity].iloc[0]
        class_row.update({
            'F1_Score': metrics['F1_Score'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'Accuracy': metrics['Accuracy']
        })
        
        # Add baseline metrics
        class_row.update({
            f'Random_{k}': v for k, v in random_multiclass['class_summary_prev']
            .loc[random_multiclass['class_summary_prev']['Class'] == activity].iloc[0].items()
            if k not in ['Class']
        })
        
        class_row.update({
            f'ZeroR_{k}': v for k, v in zero_rate_baseline['class_metrics']
            .loc[zero_rate_baseline['class_metrics']['Class'] == activity].iloc[0].items()
            if k not in ['Class', 'Prevalence']
        })
        
        results.append(class_row)
    
    return pd.DataFrame(results)
