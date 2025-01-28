import pandas as pd
import numpy as np
from pathlib import Path
from MainScript import BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS
from FeatureSelectionFunctions import clean_training_data

def remove_classes(BASE_PATH, DATASET_NAME, TRAINING_SET, TARGET_ACTIVITIES):
    """
    Highly custom function for removing speciifc behaviours from the datasets.
    To create the Open Set Test conditions.
    """

    input_path = Path(BASE_PATH) / "Data" / "Feature_data" / f"{DATASET_NAME}_other_features.csv"
    df = pd.read_csv(input_path)

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

def create_datasets(df, clean_columns, BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS):
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

    # clean the dataset
    df_clean = df[clean_columns]
    df_clean = df_clean.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # modify the data for the right model condition
    if MODEL_TYPE.lower() == 'binary' or MODEL_TYPE.lower() == 'oneclass':
        # Create activity column once using numpy where
        for behaviour in TARGET_ACTIVITIES:
            print(f"processing data for {behaviour}")
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
            elif behaviour == "OtherActivity":
                # Use numpy where for vectorized operation
                activity_column = np.where(df_clean['Activity'].isin(TARGET_ACTIVITIES), 
                                        df_clean['Activity'], 
                                        "Other")
                
                behaviour_df = df_clean.copy(deep=False)
                behaviour_df['Activity'] = activity_column
                
                save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_{MODEL_TYPE}_Other.csv")
                behaviour_df.to_csv(save_path, index=False)

    return(clean_columns)

if __name__ == "__main__":
    df = remove_classes(BASE_PATH, DATASET_NAME, TRAINING_SET, TARGET_ACTIVITIES)
    save_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_{TRAINING_SET}_other_features_subset.csv")
    df.to_csv(save_path, index=False)
    
    clean_columns = clean_training_data(training_data=df, corr_threshold = 0.9)
    # next function will create and save the dataframes]
    create_datasets(df, clean_columns, BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SETS)
    
    # clean and save the test set as well, unless it has already been done
    file_path = Path(f"{BASE_PATH}/Data/Split_data/{DATASET_NAME}_test_features_cleaned.csv")
    if file_path.exists is False:
        
        input_path_test = Path(BASE_PATH) / "Data" / "Feature_data" / f"{DATASET_NAME}_test_features.csv"
        df_test = pd.read_csv(input_path_test)
        df_clean = df_test[clean_columns]
        df_test_clean = df_clean.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        # save this
        df_test_clean.to_csv(file_path, index=False)
