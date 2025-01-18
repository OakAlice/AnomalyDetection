import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
import random
from dataclasses import dataclass


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def update_feature_data(data: pd.DataFrame, multi: str) -> pd.DataFrame:
    """Apply column selection changes to data."""
    cols_to_remove = ["Activity", "GeneralisedActivity", "OtherActivity"]
    
    if multi == "OtherActivity":
        col_to_rename = "OtherActivity"
    elif multi == "GeneralisedActivity":
        col_to_rename = "GeneralisedActivity" 
    elif multi == "Activity":
        col_to_rename = "Activity"
        
    # Remove other activity columns and rename target column
    data = data.drop(columns=[col for col in cols_to_remove if col != col_to_rename])
    data = data.rename(columns={col_to_rename: "Activity"})
    
    return data


def adjust_activity(data: pd.DataFrame, model: str, activity: str) -> pd.DataFrame:
    """Adjust activity labels based on model type."""
    data = data.copy()
    data["Activity"] = data["Activity"].apply(lambda x: activity if x == activity else "Other")
    data = data[data["Activity"].isin([activity, "Other"])]
    return data


def ensure_activity_representation(validation_data: pd.DataFrame, model: str, 
                                 activity: str, feature_data: pd.DataFrame,
                                 validation_proportion: float,
                                 retries: int = 10) -> pd.DataFrame:
    """Ensure target activity is represented in validation data."""
    retry_count = 0
    while (validation_data["Activity"] == activity).sum() == 0 and retry_count < retries:
        retry_count += 1
        print(f"{activity} not represented in validation fold. Retrying... (Attempt {retry_count})")
        
        unique_ids = feature_data["ID"].unique()
        test_ids = random.sample(list(unique_ids), 
                               int(len(unique_ids) * validation_proportion))
        validation_data = feature_data[feature_data["ID"].isin(test_ids)]
        validation_data = adjust_activity(validation_data, model, activity)
        
    if retry_count == retries:
        raise ValueError(f"Unable to find valid validation split after {retries} attempts.")
        
    return validation_data


def balance_data(data: pd.DataFrame, activity: str) -> pd.DataFrame:
    """Balance dataset by undersampling majority classes."""
    activity_count = len(data[data["Activity"] == activity]) // len(data["Activity"].unique())
    return data.groupby("Activity").apply(
        lambda x: x.sample(n=min(len(x), activity_count))
    ).reset_index(drop=True)


def split_data(model: str, activity: str, balance: str, 
               feature_data: pd.DataFrame, validation_proportion: float
               ) -> Dict[str, pd.DataFrame]:
    """Split data into training and validation sets."""
    unique_ids = feature_data["ID"].unique()
    test_ids = random.sample(list(unique_ids), 
                           int(len(unique_ids) * validation_proportion))
    
    training_data = feature_data[~feature_data["ID"].isin(test_ids)]
    validation_data = feature_data[feature_data["ID"].isin(test_ids)]
    
    if model == "OCC":
        training_data = training_data[training_data["Activity"] == activity]
        if balance == "stratified_balance":
            validation_data = balance_data(validation_data, activity)
            
    elif model == "Binary":
        if balance == "stratified_balance":
            validation_data = balance_data(validation_data, activity)
            training_data = balance_data(training_data, activity)
            
    if model != "Multi":
        training_data = adjust_activity(training_data, model, activity)
        validation_data = adjust_activity(validation_data, model, activity)
        validation_data = ensure_activity_representation(validation_data, model, 
                                                       activity, feature_data,
                                                       validation_proportion)
        
    if model == "OCC":
        training_data = training_data[training_data["Activity"] == activity]
        
    return {
        "training_data": training_data,
        "validation_data": validation_data
    }


def save_best_params(data_name: str, model_type: str, activity: str,
                    elapsed_time: List[float], results: Dict) -> pd.DataFrame:
    """Save best parameters from model tuning."""
    features = ", ".join(set(results["Pred"][
        results["History"]["Value"].tolist().index(results["Best_Value"])
    ]))
    
    return pd.DataFrame({
        "data_name": [data_name],
        "model_type": [model_type], 
        "behaviour_or_activity": [activity],
        "elapsed": [elapsed_time[2]],
        "system": [elapsed_time[1]],
        "user": [elapsed_time[0]],
        "nu": [results["Best_Par"].get("nu")],
        "gamma": [results["Best_Par"].get("gamma")],
        "kernel": [results["Best_Par"].get("kernel")],
        "number_trees": [results["Best_Par"].get("number_trees")],
        "number_features": [results["Best_Par"].get("number_features")],
        "Best_Value": [results["Best_Value"]],
        "Selected_Features": [features]
    })


def save_best_params_RF(data_name: str, model_type: str, activity: str,
                       elapsed_time: List[float], results: Dict) -> pd.DataFrame:
    """Save best parameters from Random Forest tuning."""
    features = ", ".join(set(results["Pred"][
        results["History"]["Value"].tolist().index(results["Best_Value"])
    ]))
    
    return pd.DataFrame({
        "data_name": [data_name],
        "model_type": [model_type],
        "behaviour_or_activity": [activity],
        "elapsed": [elapsed_time[2]],
        "system": [elapsed_time[1]],
        "user": [elapsed_time[0]],
        "nodesize": [results["Best_Par"]["nodesize"]],
        "n_trees": [results["Best_Par"]["n_trees"]],
        "Best_Value": [results["Best_Value"]],
        "Selected_Features": [features]
    })


def save_results(results_list: List[pd.DataFrame], file_path: str) -> None:
    """Save results to CSV file."""
    results_df = pd.concat(results_list, ignore_index=True)
    results_df.to_csv(file_path, index=False)


def undersample(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Undersample majority classes."""
    class_counts = data[target_col].value_counts()
    min_count = class_counts.min()
    
    return data.groupby(target_col).apply(
        lambda x: x.sample(n=min_count)
    ).reset_index(drop=True)


def clean_dataset(data: pd.DataFrame, activity_col: str = "Activity") -> pd.DataFrame:
    """Clean dataset by removing invalid values."""
    data = data.dropna()
    
    features = data.drop(columns=[activity_col])
    target = data[activity_col]
    
    valid_rows = features.notna().all(axis=1) & np.isfinite(features).all(axis=1)
    
    if not valid_rows.any():
        raise ValueError("No valid rows remaining after removing NA/NaN/Inf values")
        
    clean_features = features[valid_rows]
    clean_target = target[valid_rows]
    
    result = pd.concat([clean_features, 
                       pd.Series(clean_target, name="Activity").astype("category")],
                      axis=1)
    
    return result
