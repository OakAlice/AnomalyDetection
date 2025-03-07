"""
Open Set Recognition for Animal Behaviour Classification from Accelerometer Data

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

target_activities = {
    "Vehkaoja_Dog": ["Walking", "Eating", "Shaking", "Lying chest"],
    "Ladds_Seal": ["swimming", "still", "chewing"],
    "Ferdinandy_Dog": ["walk", "eat", "lay"]  # only 3 target activities
    }

def main():
    # Import and run the create_datasets module
    import sys
    import os
    from pathlib import Path
    from itertools import product

    # import HpoOptimisation
    # import CreateDatasets
    import TrainModel
    import TestModelOpen
    # import PlotResults
    import CombineResults
    
    # Add the project root directory to Python path
    sys.path.append(BASE_PATH)

    DATASET_NAME = "Ferdinandy_Dog" 
    TARGET_ACTIVITIES = target_activities[DATASET_NAME]

    # parameters for the experiments
    FOLDS = [1, 2, 3, 4, 5]
    TRAINING_SETS = ['some', 'target', 'all']
    MODEL_TYPES = ['multi', 'binary', 'oneclass']
    BEHAVIOUR_SETS = ['Other', 'Activity']
    THRESHOLD_OPTIONS = [False, True]

    # Remove all files in Metrics folders for each fold
    for fold in FOLDS:
        metrics_path = Path(f"{BASE_PATH}/Output/fold_{fold}/Testing/Metrics/")
        if metrics_path.exists():
            print(f"Removing files in {metrics_path}")
            for file in metrics_path.glob('*.csv'):
                file.unlink()

    for FOLD in FOLDS:
        for TRAINING_SET in TRAINING_SETS:
            for MODEL_TYPE in MODEL_TYPES:
                
                if MODEL_TYPE == 'multi':
                    for BEHAVIOUR_SET in BEHAVIOUR_SETS:
                        if BEHAVIOUR_SET.lower() == 'activity':
                            for THRESHOLDING in THRESHOLD_OPTIONS:
                                print(f"Thresholding: {THRESHOLDING}")
                                # train the models the same
                                TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, 
                                                FOLD=FOLD)
                                if THRESHOLDING:
                                    TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, 
                                                REASSIGN_LABELS=True, FOLD=FOLD)
                                else:
                                    TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING, 
                                                REASSIGN_LABELS=True, FOLD=FOLD)
                                
                        else: # BEHAVIOUR_SET == 'Other'
                            TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING = False, 
                                                FOLD=FOLD)

                            TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                            TARGET_ACTIVITIES, BEHAVIOUR_SET, THRESHOLDING = False, 
                                            REASSIGN_LABELS=True, FOLD=FOLD)

                else:
                    # divide and conquer methods
                    TrainModel.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                                TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING = False, 
                                                FOLD=FOLD)
                    TestModelOpen.main(BASE_PATH, DATASET_NAME, TRAINING_SET, MODEL_TYPE, 
                                            TARGET_ACTIVITIES, BEHAVIOUR_SET = None, THRESHOLDING = False, 
                                            REASSIGN_LABELS=True, FOLD=FOLD)

    # First run the comparisons to generate data
    CombineResults.main(BASE_PATH)

    # PlotResults.main(BASE_PATH, "Ferdinandy_Dog")

    # plot the volume to performance graph
    # import PlotVolumePerformance
    # PlotVolumePerformance.main(BASE_PATH)

    # import PlotFullMulticlass
    # PlotFullMulticlass.generate_full_class_data(BASE_PATH, target_activities, FOLD = 2)
    # PlotFullMulticlass.main(BASE_PATH, target_activities, FOLD = 2)

if __name__ == "__main__":
    main()
