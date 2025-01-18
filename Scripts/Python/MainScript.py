#!/usr/bin/env python3
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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
# dataset_name = "Ladds_Seal"
dataset_name = "Vehkaoja_Dog"
base_path = "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
# base_path = "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"

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
    },
    "Anguita_Human": {
        "sample_rate": None,  # not sure
        "window_length": None,  # not sure
        "overlap_percent": None,  # not sure
        "target_activities": ["WALKING", "SITTING", "STANDING"]
    }
}

# Extract settings for current dataset
settings = window_settings[dataset_name]
sample_rate = settings["sample_rate"]
window_length = settings["window_length"]
overlap_percent = settings["overlap_percent"]
target_activities = settings["target_activities"]

# Global Variables
ML_METHOD = "SVM"  # or "Tree"
TRAINING_SETS = ["all", "some", "target"]  # behaviors that appear in training set
TRAINING_SET = ["all"]
ALL_AXES = ["Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z"]
LABEL_COLUMNS = ["Activity", "Time", "ID"]
TEST_PROPORTION = 0.2
VALIDATION_PROPORTION = 0.2
FEATURES_TYPE = ["timeseries", "statistical"]
BALANCE = "stratified_balance"

# Import custom functions
from functions.feature_generation import *
from functions.feature_selection import *
from functions.svm_model_tuning import *
from functions.tree_model_tuning import *
from functions.performance_calculation import *
from functions.other_functions import *
from functions.plot_functions import *

def main():
    # All preprocessing was handled in R

    # Hyperparameter Optimization, training and testing
    if ML_METHOD == "SVM":
        exec(open(os.path.join(base_path, "Scripts", "SVMHpoOptimisation.py")).read())
        exec(open(os.path.join(base_path, "Scripts", "SVMTrainBestModels.py")).read())
        exec(open(os.path.join(base_path, "Scripts", "SVMTestBestModels.py")).read())
    else:  # Tree
        exec(open(os.path.join(base_path, "Scripts", "TreeHpoOptimisation.py")).read())
        exec(open(os.path.join(base_path, "Scripts", "TreeTrainBestModels.py")).read())
        exec(open(os.path.join(base_path, "Scripts", "TreeTestBestModels.py")).read())

    # Results Visualization and Comparison
    exec(open(os.path.join(base_path, "Scripts", "PlottingPerformance.py")).read())
    exec(open(os.path.join(base_path, "Scripts", "PlotPredictions.py")).read())

if __name__ == "__main__":
    main()