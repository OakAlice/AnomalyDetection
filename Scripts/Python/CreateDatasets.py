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
        saves both the target training data types as well as the cleaned test data
    """

    # read in the data from the right training data condition
    input_path = Path(BASE_PATH) / "Data" / "Feature_data" / f"{DATASET_NAME}_{TRAINING_SET}_other_features.csv"
    df = pd.read_csv(input_path)

    # clean the dataset
    clean_columns = clean_training_data(training_data=df, corr_threshold = 0.9)
    df_clean = df[clean_columns]
    df_clean = df_clean.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # modify the data for the right model condition
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        # Create activity column once using numpy where
        for behaviour in TARGET_ACTIVITIES:
            activity_column = np.where(df_clean['Activity'] == behaviour, behaviour, "Other")
            
            behaviour_df = df_clean.copy(deep=False)
            behaviour_df['Activity'] = activity_column
            
            # Save the dataset
            save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_{behaviour}.csv")
            behaviour_df.to_csv(save_path, index=False)
    else:
        for behaviour in BEHAVIOUR_SETS:
            if behaviour == "Activity":
                # Just save the original dataframe without copying
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_Activity.csv")
                df_clean.to_csv(save_path, index=False)
            elif behaviour == "OtherActivity":
                # Use numpy where for vectorized operation
                activity_column = np.where(df_clean['Activity'].isin(TARGET_ACTIVITIES), 
                                        df_clean['Activity'], 
                                        "Other")
                
                behaviour_df = df_clean.copy(deep=False)
                behaviour_df['Activity'] = activity_column
                
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_Other.csv")
                behaviour_df.to_csv(save_path, index=False)
    
    # clean and save the test set as well
    input_path_test = Path(BASE_PATH) / "Data" / "Feature_data" / f"{DATASET_NAME}_test_features.csv"
    df_test = pd.read_csv(input_path_test)
    df_clean = df_test[clean_columns]
    df_test_clean = df_clean.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # save this
    save_test_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_test_features_cleaned.csv")
    df_test_clean.to_csv(save_test_path, index=False)

if __name__ == "__main__":
    create_datasets(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS)