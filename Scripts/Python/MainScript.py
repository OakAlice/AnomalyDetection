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
DATASET_NAME = "Ferdinandy_Dog"
BASE_PATH = "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
# BASE_PATH = "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"

# run variables
MODEL_TYPE = "Binary" # "Binary", "Multi", "OneClass"
THRESHOLDING = False # False or number like 0.5 
BEHAVIOUR_SETS = ["Activity", "Other", "Generalised"]
BEHAVIOUR_SET = "Other" # 'Activity', 'Generalised'
TRAINING_SET = "all" # 'all', 'some', 'target'

# Window settings for each dataset
window_settings = {
    "Vehkaoja_Dog": {
        "target_activities": ["Walking", "Eating", "Shaking", "Lying chest"]
    },
    "Ladds_Seal": {
        "target_activities": ["swimming", "still", "chewing", "facerub"]
    },
    "Ferdinandy_Dog": {
        "target_activities": ["walk", "eat", "lay"] # only 3 target activities
    }
}

# Extract settings for current dataset
settings = window_settings[DATASET_NAME]
TARGET_ACTIVITIES = settings["target_activities"]

# Global Variables
ALL_AXES = ["Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z"]
LABEL_COLUMNS = ["Activity", "Time", "ID"]
TEST_PROPORTION = 0.2
VALIDATION_PROPORTION = 0.2
BALANCE = "stratified_balance"