# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data 4                ####
# ---------------------------------------------------------------------------
# Testing highest performing hyperparmeters on the test set

base_path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
source(file.path(base_path, "Scripts", "SetUp.R"))

# load in the best parameters (presuming you made them into a csv)
hyperparamaters <- fread(file.path(base_path, "Output", "OptimalHyperparameters.csv"))
OCC_hyperparameters <- hyperparamaters %>% filter(data_name == dataset_name,  model_type == "OCC")
Multi_hyperparameters <- hyperparamaters %>% filter(data_name == dataset_name, model_type == "Multi")
  
# OCC models ----------------------------------------------------------------
for (i in length(OCC_hyperparameters$activity)){

  # define the row you want to test
  parameter_row <- OCC_hyperparameters[i,]
  
  # Load in data
  training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv")))
  training_feature_data <- training_data %>% select(-c("OtherActivity", "GeneralisedActivity"))  
  
  # make a SVM with training data
  training_feature_data <- training_feature_data %>% mutate(Activity = ifelse(Activity == parameter_row$activity, Activity, "Other"))
  selected_feature_data <- featureSelection(training_feature_data, parameter_row$number_trees, parameter_row$number_features)
  target_selected_feature_data <- selected_feature_data[Activity == as.character(parameter_row$activity),!label_columns, with = FALSE]
  
  # custom line for the Shaking dog data
  # target_selected_feature_data <- target_selected_feature_data %>% select(-"peak_freq_Accelerometer.Z")
  
  # create the optimal SVM 
    optimal_single_class_SVM <-
      do.call(
        svm,
        list(
          target_selected_feature_data,
          y = NULL,
          type = 'one-classification',
          nu = parameter_row$nu,
          scale = TRUE,
          kernel = parameter_row$kernel,
          gamma = parameter_row$gamma
        )
      )
    
  # save this model
  model_path <- file.path(base_path, "Output", "Models", paste0(parameter_row$data_name, "_", parameter_row$activity, "_final_model.rda"))
  save(optimal_single_class_SVM, file = model_path)
  
  # load in the test data
  testing_data <- fread(file = file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv")))
  testing_feature_data <- testing_data %>% select(-c("OtherActivity", "GeneralisedActivity"))

  # test the performance
  # I also wrote it to test training data and randomised test data. Check documentation.
  testing_results <- finalModelPerformance(mode = "testing",
                                          training_data = target_selected_feature_data,
                                          optimal_model = optimal_single_class_SVM,
                                          testing_data = testing_feature_data,
                                          target_activity = parameter_row$activity,
                                          balance = TRUE)
  
  print(testing_results)
}


# Multi-class models ------------------------------------------------------
for (i in length(Multi_hyperparameters$activity)){
  
  # i <- 1
  # define the row you want to test
  parameter_row <- Multi_hyperparameters[i,]
  
  training_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv")))
  testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv")))
  
  # select the right column for the testing activity based on multi, and remove the others
  training_feature_data <- update_feature_data(training_data, parameter_row$activity)
  training_feature_data <- training_feature_data[!Activity == ""]
  
  testing_feature_data <- update_feature_data(testing_data, parameter_row$activity)
  testing_feature_data <- testing_feature_data[!Activity == ""]
  
  # run this manually to get per class metrics
  baseline_results <- baselineMultiClass(dataset_name,
                                         condition = parameter_row$activity,
                                         training_data = training_feature_data,
                                         testing_data = testing_feature_data,
                                         number_trees = parameter_row$number_trees,
                                         number_features = parameter_row$number_features,
                                         kernel = parameter_row$kernel,
                                         gamma = parameter_row$gamma
                                         )
  
  print(baseline_results)
}

