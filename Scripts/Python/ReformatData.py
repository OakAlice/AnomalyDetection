# reformat the Ferdinandy dataset to be a csv
# read in the dat format

import pandas as pd
from pathlib import Path
from MainScript import BASE_PATH

# Read the raw data file
# with open(Path(f"{BASE_PATH}/Data/ferdinandyb/smalldog.dat"), 'r') as file:
#     lines = file.readlines()

# Process each line into a list of values
# processed_data = []
# for line in lines:
#     # Split by comma and convert to list
#     values = line.strip().split(',')
#     processed_data.append(values)

# # Create DataFrame with appropriate columns
# # Adjust these column names based on your actual data structure
# columns = ['size', 'id', 'start_time', 'end_time', 'activity'] + [f'feature_{i+1}' for i in range(len(processed_data[0])-5)]
# df = pd.DataFrame(processed_data, columns=columns)

# # Save to csv
# df.to_csv(Path(f"{BASE_PATH}/Data/ferdinandyb/smalldog.csv"), index=False)

# # read in the 3 csvs and concatenate them
# df = pd.concat([pd.read_csv(Path(f"{BASE_PATH}/Data/ferdinandyb/smalldog.csv")),
#                 pd.read_csv(Path(f"{BASE_PATH}/Data/ferdinandyb/mediumdog.csv")),
#                 pd.read_csv(Path(f"{BASE_PATH}/Data/ferdinandyb/largedog.csv"))])

# # save the concatenated dataframe to a new csv
# df.to_csv(Path(f"{BASE_PATH}/Data/Ferdinandy_Dog.csv"), index=False)


# divide into training and testing 
# select 20% of the ID for the test set
# data = pd.read_csv(Path(f"{BASE_PATH}/Data/Ferdinandy_Dog.csv"))
# test_ids = data['ID'].unique()[:int(len(data['ID'].unique())*0.2)]

# # save the test set to a new csv
# data[data['ID'].isin(test_ids)].to_csv(Path(f"{BASE_PATH}/Data/Feature_data/Ferdinandy_Dog_test_features.csv"), index=False)

# # save the training set to a new csv
# data[~data['ID'].isin(test_ids)].to_csv(Path(f"{BASE_PATH}/Data/Feature_data/Ferdinandy_Dog_all_other_features.csv"), index=False)

# Define the path properly with Path object and proper string formatting
path = Path(r"C:\Users\oaw001\OneDrive - University of the Sunshine Coast\AnomalyDetection\Data\Split_data")

# Get all CSV files containing 'Ferdinandy_Dog' in the name
ferdinandy_files = list(path.glob("*Ferdinandy_Dog*.csv"))

# Process each file
for file_path in ferdinandy_files:
    print(file_path)
    # Read the CSV file
    df = pd.read_csv(file_path, low_memory=False)
    
    # Remove specified columns if they exist
    columns_to_remove = ['size', 'group', 'EndTime']
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)
    
    # Save the modified DataFrame back to the same file
    df.to_csv(file_path, index=False)