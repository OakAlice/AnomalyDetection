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

    # HpoOptimisation.append_files(BASE_PATH)

    CompareConditions.plot_auc_comparison(BASE_PATH)

    # and then do the final model run
    DATASET_NAME = "Vehkaoja_Dog"
    TARGET_ACTIVITIES = target_activities[DATASET_NAME]
    TRAINING_SET = 'some'
    MODEL_TYPE = 'multi'
    BEHAVIOUR_SET = 'Other'
    THRESHOLDING = False

    TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING = False)
    TestModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                   TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
    

    
    #             #HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)

    #             #HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
    #             #TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
                
    #             # Define test configurations based on MODEL_TYPE
    #             test_configs = []
    #             if MODEL_TYPE == 'multi':
    #                 test_configs = [
    #                     ('Activity', 0.5),
    #                     ('Activity', False),
    #                     ('Other', False)
    #                 ]
    #             else:
    #                 test_configs = [('Activity', False)]
                
    #             # Run tests for each configuration
    #             for BEHAVIOUR_SET, THRESHOLDING in test_configs:
    #                 TestModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                              TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING)
    
    # for DATASET_NAME in ['Ferdinandy_Dog', "Vehkaoja_Dog"]:
    #     TARGET_ACTIVITIES = target_activities[DATASET_NAME]
    #     CompareConditions.main(BASE_PATH, DATASET_NAME, TARGET_ACTIVITIES)

    # Call the plotting function after the comparison loop
    CompareConditions.plot_auc_comparison(BASE_PATH)
    CompareConditions.elapsed_time(BASE_PATH)

if __name__ == "__main__":
    main()
