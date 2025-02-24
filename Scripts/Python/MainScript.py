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
FOLD = 2 # 0, 1, 2, 3, 4
# TRAINING_SET = "some" # 'all', 'some', 'target'

target_activities = {
    "Vehkaoja_Dog": ["Walking", "Eating", "Shaking", "Lying chest"],
    "Ladds_Seal": ["swimming", "still", "chewing"],
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
    import TestModelOpen
    import PlotResults
    import TestModelClosed
    import CombineResults
    
    # Add the project root directory to Python path
    sys.path.append(BASE_PATH)

    # HpoOptimisation.append_files(BASE_PATH)

    # and then do the final model run
    # for FOLD in [1, 2, 3, 4, 5]:
    #     for DATASET_NAME in ["Ferdinandy_Dog"]:
    #         TARGET_ACTIVITIES = target_activities[DATASET_NAME]

    #         # CreateDatasets.main(DATASET_NAME, TARGET_ACTIVITIES, FOLD)

    #         for TRAINING_SET in ['some', 'target', 'all']:
    #             for MODEL_TYPE in ['multi']:
                
    #                 print(f"Running {DATASET_NAME} with training set {TRAINING_SET} and {MODEL_TYPE} model for fold {FOLD}")
                    
    #                 if MODEL_TYPE == 'multi':
    #                     for BEHAVIOUR_SET in ['Activity', 'Other']:
    #                         if BEHAVIOUR_SET == 'Activity':
    #                             for THRESHOLDING in [False]:
    #                                 print(f"thresholding: {THRESHOLDING}")
    #                                 # HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                                 #                 TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
    #                                 # HpoOptimisation.append_files(BASE_PATH, FOLD)
    #                                 # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                                 #                 TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
    #                                 TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                                                     TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=True, FOLD = FOLD)
    #                                 if THRESHOLDING is False:
    #                                     TestModelClosed.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BEHAVIOUR_SET, THRESHOLDING, FOLD)
    #                         else:
    #                             THRESHOLDING = False
    #                             # HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                             #                     TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
    #                             # HpoOptimisation.append_files(BASE_PATH, FOLD)
    #                             # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                             #                  TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
    #                             # TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                             #                  TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=True, FOLD = FOLD)
    #                 else:
    #                     THRESHOLDING = False
                        # HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                        #                             TARGET_ACTIVITIES, behaviour_set = None, thresholding= False, fold= FOLD)
                        # HpoOptimisation.append_files(BASE_PATH, FOLD)
                        # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                        #                   TARGET_ACTIVITIES, BEHAVIOUR_SET= None, THRESHOLDING = False, FOLD = FOLD)
                    
                        # TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                        #               TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING =False, REASSIGN_LABELS=True, FOLD = FOLD)
       #  HpoOptimisation.append_files(BASE_PATH, FOLD)


    # DATASET_NAME = "Ferdinandy_Dog"
    # TARGET_ACTIVITIES = target_activities[DATASET_NAME] 
    # MODEL_TYPE = "multi"
    # BEHAVIOUR_SET = "Other"
    # THRESHOLDING = False

    # for FOLD in [1, 2, 3, 4, 5]:
    #     for TRAINING_SET in ['all', 'some', 'target']:
    #         for MODEL_TYPE in ['oneclass', 'binary', 'multi']:
    #             if MODEL_TYPE == 'multi':
    #                 for BEHAVIOUR_SET in ['Activity', 'Other']:
    #                     if BEHAVIOUR_SET == 'Activity':
    #                         for THRESHOLDING in [False, True]: 
    #                             print(FOLD)
    #                             TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                             TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=True, FOLD = FOLD)
    #                     else:
    #                         TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                         TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING = False, REASSIGN_LABELS=True, FOLD = FOLD)
    #             else:    
    #                 TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #                     TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING = False, REASSIGN_LABELS=True, FOLD = FOLD)

    
    # # # First run the comparisons to generate data
    #  CombineResults.main(BASE_PATH)

    PlotResults.main(BASE_PATH, "Ferdinandy_Dog")

    # plot the volume to performance graph
    # import PlotVolumePerformance
    # PlotVolumePerformance.main(BASE_PATH)

    # import PlotFullMulticlass
    # PlotFullMulticlass.generate_full_class_data(BASE_PATH, target_activities, FOLD = 2)
    # PlotFullMulticlass.main(BASE_PATH, target_activities, FOLD = 2)

if __name__ == "__main__":
    main()
