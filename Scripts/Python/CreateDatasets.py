import pandas as pd
import numpy as np
from pathlib import Path
import os  # Add this import for extra path handling compatibility
from MainScript import BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES
from FeatureSelectionFunctions import clean_training_data

def remove_classes(BASE_PATH, DATASET_NAME, TRAINING_SET, TARGET_ACTIVITIES):
    """
    Highly custom function for removing speciifc behaviours from the datasets.
    To create the Open Set Test conditions.
    """
    input_path = Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_train.csv"
    # Add low_memory=False to avoid DtypeWarning
    df = pd.read_csv(input_path, low_memory=False)

    # Define metadata columns explicitly and ensure rest are numeric
    metadata_columns = ['Activity', 'ID', 'Time']
    numeric_columns = df.columns.difference(metadata_columns)
    # Convert only columns that are actually numeric
    for col in numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            print(f"Could not convert column {col} to numeric")
            numeric_columns = numeric_columns.drop(col)

    if TRAINING_SET == 'some':
        # remove the uncommon behaviours
        counts = df['Activity'].value_counts()
        print(counts)
        # Select top 50% of activities by frequency
        n_activities = len(counts)
        common_activities = counts.index[:n_activities//2].tolist()
            
        # Combine common activities with target activities and remove duplicates
        common_activities = list(set(common_activities + TARGET_ACTIVITIES))

        # only keep the common activities (also target activities if not included in top counts)
        df = df[df['Activity'].isin(common_activities)]
    
    elif TRAINING_SET == 'target':
        # only keep the target activities
        df = df[df['Activity'].isin(TARGET_ACTIVITIES)]

    else:
        print("keep everything, nothing to change")

    return df

def create_datasets(df, clean_columns, BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES):
    """
    Create and save datasets based on specified parameters
    
    Args:
        DATASET_NAME (str): Name of the dataset
        TRAINING_SET (str): Training set identifier
        MODEL_TYPE (str): Type of model ('binary', 'oneclass', or 'multi')
        BASE_PATH (str): Base path for data
        TARGET_ACTIVITIES (list): List of target activities
    
    Returns:
        saves both the target training data types as well as the cleaned test data
    """
    # Define metadata columns to preserve
    metadata_columns = ['ID', 'Time'] # activity already in there
    
    # Add metadata columns to clean_columns
    all_columns = list(clean_columns) + metadata_columns
    
    # clean the dataset but keep metadata
    df_clean = df[all_columns]
    df_clean = df_clean.dropna(subset=clean_columns).replace([np.inf, -np.inf], np.nan).dropna(subset=clean_columns)

    # modify the data for the right model condition
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        # Create activity column once using numpy where
        for behaviour in TARGET_ACTIVITIES:
            print(f"processing data for {behaviour}")
            activity_column = np.where(df_clean['Activity'] == behaviour, behaviour, "Other")

            print(activity_column)
            
            behaviour_df = df_clean.copy(deep=False)
            behaviour_df['Activity'] = activity_column
            
            # Save the dataset
            save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv")
            behaviour_df.to_csv(save_path, index=False)
    else:
        for behaviour in ["Activity", "Other"]:
            if behaviour == "Activity":
                # Save the original dataframe
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_Activity.csv")
                df_clean.to_csv(save_path, index=False)
            elif behaviour == "Other":
                # Use numpy where for vectorized operation
                activity_column = np.where(df_clean['Activity'].isin(TARGET_ACTIVITIES), 
                                        df_clean['Activity'], 
                                        "Other")
                
                behaviour_df = df_clean.copy(deep=False)
                behaviour_df['Activity'] = activity_column
                
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_Other.csv")
                behaviour_df.to_csv(save_path, index=False)

    return(clean_columns)

def split_test_data(BASE_PATH, DATASET_NAME):
    test_path = Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_test.csv"
    train_path = Path(BASE_PATH) / "Data" / "Split_data" / f"{DATASET_NAME}_train.csv"
    
    if test_path.exists():
        print(f"Test data already split for {DATASET_NAME}. If you change it, you will need to delete the file and also redo everything.")
    else:
        # Load the full dataset
        features_path = Path(BASE_PATH) / "Data" / "Feature_data" / f"{DATASET_NAME}_features.csv"
        df_full = pd.read_csv(features_path, low_memory=False)
        
        # Select all data from 20% of individuals for testing
        unique_ids = df_full['ID'].unique()
        test_ids = np.random.choice(unique_ids, size=int(len(unique_ids) * 0.2), replace=False)
        
        # Split into test and train
        df_test = df_full[df_full['ID'].isin(test_ids)]
        df_train = df_full[~df_full['ID'].isin(test_ids)]
        
        # Save both datasets
        df_test.to_csv(test_path, index=False)
        df_train.to_csv(train_path, index=False)

def main():
    # split out the test data
    split_test_data(BASE_PATH, DATASET_NAME)
    
    # generate the individual datasets used in each subsequent model design
    for TRAINING_SET in ['all', 'some', 'target']:
        df = remove_classes(BASE_PATH, DATASET_NAME, TRAINING_SET, TARGET_ACTIVITIES)
            
        for MODEL_TYPE in ['binary', 'oneclass', 'multi']:
            clean_columns = clean_training_data(training_data=df, corr_threshold = 0.9)
            # next function will create and save the dataframes]
            create_datasets(df, clean_columns, BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES)

if __name__ == "__main__":
    main()