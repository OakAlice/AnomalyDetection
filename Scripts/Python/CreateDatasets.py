import pandas as pd
import numpy as np
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS
from FeatureSelectionFunctions import clean_training_data

def create_datasets(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS):
    """
    Create and save datasets based on specified parameters
    
    Args:
        DATASET_NAME (str): Name of the dataset
        TRAINING_SET (str): Training set identifier
        MODEL_TYPE (str): Type of model ('binary', 'oneclass', or 'multi')
        BASE_PATH (str): Base path for data
        TARGET_ACTIVITIES (list): List of target activities
        BEHAVIOUR_SETS (list): List of behaviour sets
    
    Returns:
        dict: Dictionary containing the processed dataframes
    """

    # read in the data from the right training data condition
    input_path = Path(BASE_PATH) / "Data" / "Feature_data" / f"{DATASET_NAME}_{TRAINING_SET}_other_features.csv"
    df = pd.read_csv(input_path)

    # clean the dataset
    clean_columns = clean_training_data(training_data=df, corr_threshold = 0.9)
    df_clean = df[clean_columns]
    df_clean = df_clean.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # modify the data for the right model condition
    behaviour_dfs = {}  # Dictionary to store all behavior dataframes
    
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        for behaviour in TARGET_ACTIVITIES:
            behaviour_dfs[behaviour] = df_clean.copy()
            behaviour_dfs[behaviour]['Activity'] = df_clean['Activity'].apply(
                lambda x: behaviour if x == behaviour else "Other"
            )
            # Save the dataset
            save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv")
            behaviour_dfs[behaviour].to_csv(save_path, index=False)
    else:
        for behaviour in BEHAVIOUR_SETS:
            behaviour_dfs[behaviour] = df_clean.copy()
            if behaviour == "Activity":
                # Keep the dataframe as is with all original activities
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_full.csv")
            elif behaviour == "OtherActivity":
                # Keep TARGET_ACTIVITIES as they are, make all others "Other"
                behaviour_dfs[behaviour]['Activity'] = df_clean['Activity'].apply(
                    lambda x: x if x in TARGET_ACTIVITIES else "Other"
                )
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_other.csv")
            
            # Save the dataset
            behaviour_dfs[behaviour].to_csv(save_path, index=False)
    
    # return the dataframes and keep for later
    return behaviour_dfs

if __name__ == "__main__":
    create_datasets(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS)