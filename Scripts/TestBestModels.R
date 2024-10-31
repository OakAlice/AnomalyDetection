# Testing highest performing hyperparmeters for OCC ----------------------
# currently have to write out the best performing hyperparameters then run
# haven't got it automated yet



base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
dataset_name <- "Ladds_Seal"



library(data.table)
library(tidyverse)


# define the best parameters
target_activity <- "still"
window_length <- 1
number_trees <- 263
number_features <- 12
kernel <- "radial"
nu <- 0.058
gamma <- 0.048



# 1. OCC models -----------------------------------------------------------

# Load in data ------------------------------------------------------------
# load in the data for the right window length
testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))

# remove the other columns for the OCC models
testing_feature_data <- testing_data %>% select(-c("OtherActivity", "GeneralisedActivity"))
training_feature_data <- training_data %>% select(-c("OtherActivity", "GeneralisedActivity"))  





# make a SVM with training data
training_feature_data <- training_feature_data %>% mutate(Activity = ifelse(Activity == target_activity, Activity, "Other"))
selected_feature_data <- featureSelection(training_feature_data, number_trees, number_features)
target_selected_feature_data <- selected_feature_data[Activity == as.character(target_activity),!label_columns, with = FALSE]

# create the optimal SVM ####
  optimal_single_class_SVM <-
    do.call(
      svm,
      list(
        target_selected_feature_data,
        y = NULL,
        type = 'one-classification',
        nu = nu,
        scale = TRUE,
        kernel = kernel,
        gamma = gamma
      )
    )
  
# save this model
model_path <- file.path(base_path, "Output", "Models", paste0(target_activity, "_", dataset_name, "_model.rda"))
save(optimal_single_class_SVM, file = model_path)
  

# I also wrote it so if you change mode to training and remove 
# testing data it shows performance on the training set
# also has a random mode if want to randomise target activity
  
testing_results <- finalModelPerformance(mode = "testing",
                                          training_data = target_selected_feature_data,
                                          optimal_model = optimal_single_class_SVM,
                                          testing_data = testing_feature_data,
                                          target_activity = target_activity,
                                          balance = TRUE)
testing_results


# Testing highest performing hyperparmeters for multi----------------------
multi <- "GeneralisedActivity" # "OtherActivity", "Activity"
number_trees <- 232
number_features <- 100
kernel <- "radial"
gamma <- 0.003

if (testingMulti == TRUE){
  training_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
  testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
  
  # select the right column for the testing activity based on multi, and remove the others
  training_feature_data <- update_feature_data(training_data, multi)
  training_feature_data <- training_feature_data[!Activity == ""]
  testing_feature_data <- update_feature_data(testing_data, multi)
  testing_feature_data <- testing_feature_data[!Activity == ""]
  
  baseline_results <- baselineMultiClass(training_data = training_feature_data,
                                         testing_data = testing_feature_data,
                                         number_trees = number_trees,
                                         number_features = number_features,
                                         kernel = kernel,
                                         gamma = gamma)
  
  print(paste0(multi, " multi results:"))
  baseline_results
}