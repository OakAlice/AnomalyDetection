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
# BASE_PATH = "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
BASE_PATH = "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
        
# defaults # or run variables
DATASET_NAME = "Ferdinandy_Dog" # "Vehkaoja_Dog", "Ladds_Seal", "Ferdinandy_Dog"
# MODEL_TYPE = "Multi" # "Binary", "Multi", "OneClass"
THRESHOLDING = False # False or 0.5 
BEHAVIOUR_SET = "Activity" # 'Activity' or 'Other'
# TRAINING_SET = "some" # 'all', 'some', 'target'

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

    import HpoOptimisation
    import CreateDatasets
    import TrainModel
    import TestModel
    import CompareConditions

    
    # Add the project root directory to Python path
    sys.path.append(BASE_PATH)
    
    #             HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)
    # HpoOptimisation.append_files(BASE_PATH)

    # for DATASET_NAME in ['Ferdinandy_Dog']:
    #     TARGET_ACTIVITIES = target_activities[DATASET_NAME]
    #     for MODEL_TYPE in ['binary', 'oneclass', 'multi']:
    #         for TRAINING_SET in ['some', 'target', 'all']:

    #             if TRAINING_SET == 'all' and MODEL_TYPE == 'binary':
    #                 continue # skip the condition I haven't trained yet.

    #             if MODEL_TYPE.lower() == 'multi':
    #                 for BEHAVIOUR_SET in ['Activity', 'Other']:
    #                     # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET)

    #                     TestModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
    #             else:
    #                 # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET = None)

    #                 TestModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING = THRESHOLDING)
    
    for DATASET_NAME in ['Ferdinandy_Dog', "Vehkaoja_Dog"]:
        TARGET_ACTIVITIES = target_activities[DATASET_NAME]
        CompareConditions.main(BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES)

if __name__ == "__main__":
    main()