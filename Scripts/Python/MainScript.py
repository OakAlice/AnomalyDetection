"""
Animal Behaviour Classification using Machine Learning

This program implements a comparative analysis of machine learning approaches
for classifying animal behaviours from accelerometer data.
It compares SVM and Random Forest based systems for:
- One-Class Classification (OCC)
- Binary Classification
- Multi-class Classification
"""

import os

# Configuration
DATASET_NAME = "Vehkaoja_Dog"
# BASE_PATH = "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
BASE_PATH = "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"

# run variables
ML_METHOD = "SVM"
MODEL_TYPE = "Binary" # "Binary", "Multi", "OneClass"
THRESHOLDING = False # False or number like 0.5 
BEHAVIOUR_SETS = ["Activity", "Other", "Generalised"]
BEHAVIOUR_SET = "Other" # 'Activity', 'Generalised'
TRAINING_SET = "some" # 'all', 'some', 'target'

# Window settings for each dataset
window_settings = {
    "Vehkaoja_Dog": {
        "sample_rate": 100,
        "window_length": 1,
        "overlap_percent": 50,
        "target_activities": ["Walking", "Eating", "Shaking", "Lying chest"]
    },
    "Ladds_Seal": {
        "sample_rate": 25,
        "window_length": 1,
        "overlap_percent": 50,
        "target_activities": ["swimming", "still", "chewing", "facerub"]
    }
}

# Extract settings for current dataset
settings = window_settings[DATASET_NAME]
SAMPLE_RATE = settings["sample_rate"]
WINDOW_LENGTH = settings["window_length"]
OVERLAP_PERCENT = settings["overlap_percent"]
TARGET_ACTIVITIES = settings["target_activities"]

# Global Variables
ALL_AXES = ["Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z"]
LABEL_COLUMNS = ["Activity", "Time", "ID"]
TEST_PROPORTION = 0.2
VALIDATION_PROPORTION = 0.2
FEATURES_TYPE = ["timeseries", "statistical"]
BALANCE = "stratified_balance"

# def main():
    # All preprocessing was handled in R

    # Hyperparameter Optimization, training and testing
 #   if ML_METHOD == "SVM":
 #       exec(open(os.path.join(BASE_PATH, "Scripts", "SVMHpoOptimisation.py")).read())
 #       exec(open(os.path.join(BASE_PATH, "Scripts", "SVMTrainBestModels.py")).read())
 #       exec(open(os.path.join(BASE_PATH, "Scripts", "SVMTestBestModels.py")).read())
 #   else:  # Tree
 #       exec(open(os.path.join(BASE_PATH, "Scripts", "TreeHpoOptimisation.py")).read())
 #       exec(open(os.path.join(BASE_PATH, "Scripts", "TreeTrainBestModels.py")).read())
 #       exec(open(os.path.join(BASE_PATH, "Scripts", "TreeTestBestModels.py")).read())

    # Results Visualization and Comparison
 #   exec(open(os.path.join(BASE_PATH, "Scripts", "PlottingPerformance.py")).read())
 #   exec(open(os.path.join(BASE_PATH, "Scripts", "PlotPredictions.py")).read())

# if __name__ == "__main__":
#    main()