"""
Animal Behaviour Classification using Machine Learning

This program implements a comparative analysis of machine learning approaches
for classifying animal behaviours from accelerometer data.
It compares SVM and Random Forest based systems for:
- One-Class Classification (OCC)
- Binary Classification
- Multi-class Classification
"""

# define the variables
BASE_PATH = "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
# BASE_PATH = "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
        
# run variables
DATASET_NAME = "Vehkaoja_Dog" # "Vehkaoja_Dog", "Ladds_Seal", "Ferdinandy_Dog"
MODEL_TYPE = "Multi" # "Binary", "Multi", "OneClass"
THRESHOLDING = False # False or 0.5 
BEHAVIOUR_SET = "Activity" # 'Activity' or 'Other'
TRAINING_SET = "all" # 'all', 'some', 'target'

target_activities = {
    "Vehkaoja_Dog": ["Walking", "Eating", "Shaking", "Lying chest"],
    "Ladds_Seal": ["swimming", "still", "chewing", "facerub"],
    "Ferdinandy_Dog": ["walk", "eat", "lay"]  # only 3 target activities
    }
TARGET_ACTIVITIES = target_activities[DATASET_NAME]

def main():
    # Import and run the create_datasets module
    import sys
    import os
    
    # Add the project root directory to Python path
    sys.path.append(BASE_PATH)
    
    # from Scripts.Python import CreateDatasets
    #CreateDatasets.main()

    for TRAINING_SET in ['all', 'some', 'target']:
        for MODEL_TYPE in ['binary', 'oneclass', 'multi']:
                from Scripts.Python import HpoOptimisation
                HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)

    # from Scripts.Python import TrainModel
    # TrainModel.main()
    
    # from Scripts.Python import TestModel
    # TestModel.main()

    # test the optimal models
    # exec(open(os.path.join(BASE_PATH, "Scripts", "Python", "TestModel.py")).read())

if __name__ == "__main__":
    main()