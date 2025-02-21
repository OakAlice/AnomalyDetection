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
    import TestModelOpen
    import PlotResults
    import TestModelClosed
    import CombineResults
    
    # Add the project root directory to Python path
    sys.path.append(BASE_PATH)

    # HpoOptimisation.append_files(BASE_PATH)

    # and then do the final model run
    for FOLD in [2, 3, 4, 5]:
        for DATASET_NAME in ["Vehkaoja_Dog"]:
            TARGET_ACTIVITIES = target_activities[DATASET_NAME]
            
            CreateDatasets.main(DATASET_NAME, TARGET_ACTIVITIES, FOLD)

            for MODEL_TYPE in ['binary']: # 'binary', 'oneclass'
                for TRAINING_SET in ['some']: 

                    print(f"Running {DATASET_NAME} with training set {TRAINING_SET} and {MODEL_TYPE} model for fold {FOLD}")
                    
                    if MODEL_TYPE == 'multi':
                        for BEHAVIOUR_SET in ['Activity', 'Other']:
                            if BEHAVIOUR_SET == 'Activity':
                                THRESHOLDING = False
                                print(f"thresholding: {THRESHOLDING}")
                                # HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                #                  TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
                                # HpoOptimisation.append_files(BASE_PATH, FOLD)
                                # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                #                 TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
                                # TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                #                    TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=True, FOLD = FOLD)
                                # TestModelClosed.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BEHAVIOUR_SET, THRESHOLDING, FOLD)
                            else:
                                THRESHOLDING = False
                                # HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                #                     TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
                                # HpoOptimisation.append_files(BASE_PATH, FOLD)
                                # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                #                  TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, FOLD)
                                # TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                #                  TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, REASSIGN_LABELS=True, FOLD = FOLD)
                                # TestModelClosed.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BEHAVIOUR_SET, THRESHOLDING, FOLD)
                    else:
                        THRESHOLDING = False
                        HpoOptimisation.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                    TARGET_ACTIVITIES, behaviour_set = None, thresholding= False, fold= FOLD)
                        HpoOptimisation.append_files(BASE_PATH, FOLD)
                        TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                          TARGET_ACTIVITIES, BEHAVIOUR_SET= None, THRESHOLDING = False, FOLD = FOLD)
                    
                        TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                        TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING =False, REASSIGN_LABELS=True, FOLD = FOLD)
                        # TestModelClosed.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, BEHAVIOUR_SET = None, THRESHOLDING = False, FOLD = FOLD)
                    
       #  HpoOptimisation.append_files(BASE_PATH, FOLD)


    # DATASET_NAME = "Vehkaoja_Dog"
    # TARGET_ACTIVITIES = target_activities[DATASET_NAME] 
    # FOLD = 4
    # MODEL_TYPE = "binary"
    # BEHAVIOUR_SET = "Activity"
    # THRESHOLDING = False
    # TRAINING_SET = "some"






    # TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #     TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING = False, FOLD = FOLD)
    # TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
    #     TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING = False, REASSIGN_LABELS=True, FOLD = FOLD)




    
    # First run the comparisons to generate data
    # CombineResults.main(BASE_PATH)

    # PlotResults.main(BASE_PATH)

    # plot the volume to performance graph
    # import PlotVolumePerformance
    # PlotVolumePerformance.main(BASE_PATH)

    # import PlotFullMulticlass
    # PlotFullMulticlass.main(BASE_PATH, target_activities)

if __name__ == "__main__":
    main()
