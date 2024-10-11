# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data                  ####
# ---------------------------------------------------------------------------

# script mode
# tuning for HPO finding, testing for final validation
tuning <- FALSE
testing <- TRUE

# User Defined Variables ---------------------------------------------------
# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
dataset_name <- "Ladds_Seal"
sample_rate <- 25

# Set up ------------------------------------------------------------------
library(renv)
# commented out as I already have this
# if (file.exists(file.path(base_path, "renv.lock"))) {
#   renv::restore() # this will install all the right versions of the packages
# }

library(pacman)
p_load(
  bench, caret, data.table, e1071, future, future.apply, parallelly,
  plotly, PRROC, purrr, pROC, rBayesianOptimization,
  randomForest, tsfeatures, tidyverse, umap, zoo
)
#library(h2o) # for UMAP, but takes a while so ignore unless necessary

# load in the scripts
scripts <-
  list(
    "BaselineSVM.R",
    "FeatureGeneration.R",
    "FeatureSelection.R",
    "OtherFunctions.R",
    "ModelTuning.R",
    "CalculatePerformance.R"
  )

# Function to source scripts and handle errors
successful <- TRUE
source_script <- function(script) {
  tryCatch(
    source(file.path(base_path, "Scripts", script)),
    error = function(e) {
      successful <<- FALSE
      message(paste("Error sourcing script:", script))
    }
  )
}
walk(scripts, source_script)

# some other things I need defined globally
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")


# Create hold out test data -----------------------------------------------
move_data <- fread(file.path(base_path, "Data", paste0(dataset_name, ".csv")))

# Split Data ####
if (file.exists(file.path(
  base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")
  ))) {
  # if this has been run before, just load in the split data
  data_test <-
    fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  data_other <-
    fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")))
} else {
  # if this is the first time running code for this dataset, create hold-out test set
  unique_ids <- unique(move_data$ID)
  test_ids <-
    sample(unique_ids, ceiling(length(unique_ids) * test_proportion))
  data_test <- move_data[ID %in% test_ids]
  data_other <- move_data[!ID %in% test_ids]
  # save these
  fwrite(data_test,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  fwrite(data_other,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")
         ))
}

# Explore Data ------------------------------------------------------------
# Go into the ExploreData.Rmd RMarkdown file to find variables
# then manually specify below

target_activities <- c("swimming", "moving", "still", "chewing")
window_length <- 1 # TODO: Add that each behaviour has a different one
overlap_percent <- 0

# Feature Generation ------------------------------------------------------
if (file.exists(
  file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")
))) {
  feature_data <-
    fread(
      file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")
    ))
} else {
  
  for (id in unique(data_other$ID))  {
  dat <- data_other %>% filter(ID == id)
  
  feature_data <-
    generateFeatures(
      window_length,
      sample_rate,
      overlap_percent,
      data = dat,
      normalise = "z_scale",
      features = features_type
    )
  # save it 
  fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_other_features.csv")))
}
  # stitch all the id feature data back together
  files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
  matching_files <- grep(dataset_name, files, value = TRUE)
  
  feature_data_list <- lapply(matching_files, read.csv)
  feature_data <- do.call(rbind, feature_data_list)
  # save this as well
  fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
}

# Tuning model hyperparameters --------------------------------------------
# this section of the code iterates through hyperparameter combinations for OCC
# PR-ROC for the target class is recorded and saved in the output table

# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.01, 0.1),
  gamma = c(0.01, 0.1),
  number_trees = c(100, 500),
  number_features = c(10, 50)
)

ensure.dir(file.path(base_path, "Output"))
if (tuning == TRUE){

  for (target_activity in target_activities) {
    target_activity <- "swimming"
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = function(nu, gamma, number_trees, number_features) {
        modelTuning(
          feature_data = feature_data,  # Pass feature_data as a fixed argument
          nu = nu,
          kernel = "radial",
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
      },
      bounds = bounds,
      init_points = 5,
      n_iter = 15,
      acq = "ucb",
      kappa = 2.576 
    )
  }
}

# Testing highest performing hyperparmeters -------------------------------
# currently have to write out the best performing hyperparameters then run
# haven't got it automated yet

target_activity <- "swimming"
number_trees <- 500
number_features <- 10
kernel <- "radial"
nu <- 0.01
gamma <- 0.06

if (testing == TRUE){
  
  print("generating features for test data")
  
  # ## load in the test data and generate appropriate features ####
  if (file.exists(file.path(
    file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")
  )))) {
    # if this has been run before, just load in
    testing_feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
  } else {
    # calculate and save
    testing_data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")))
   
    for (id in unique(testing_data$ID)){
      testing_feature_data <- generateFeatures(window_length, 
                                               sample_rate, 
                                               overlap_percent, 
                                               testing_data,
                                               features_type)
      fwrite(testing_feature_data,
           file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
    }
  }
  print("generating optimal model")
  
  
  target_activity <- "chewing"
  # # make a SVM with training data
  # ## load in training data and select features and target data ####
  training_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
  selected_feature_data <- featureSelection(training_data, number_trees, number_features)
  target_selected_feature_data <- selected_feature_data[Activity == as.character(target_activity),!label_columns, with = FALSE]
  # ## create the optimal SVM ####
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
  
  print("calculating performance")
  # # calculate performance of the final model in various conditions ####
  training_results <- finalModelPerformance(mode = "training",
                                            training_data = target_selected_feature_data,
                                            optimal_model = optimal_single_class_SVM)
  print("Training data:")
  training_results
  
  testing_results <- finalModelPerformance(mode = "testing",
                                            training_data = target_selected_feature_data,
                                            optimal_model = optimal_single_class_SVM,
                                            testing_data = testing_feature_data,
                                            target_activity = target_activity)
  print("Testing data:")
  testing_results
  
  random_results <- finalModelPerformance(mode = "random",
                                           training_data = target_selected_feature_data,
                                           optimal_model = optimal_single_class_SVM,
                                           testing_data = testing_feature_data,
                                           target_activity = target_activity)
  print("Random data:")
  random_results
  
  baseline_results <- baselineMultiClass(training_data = training_data,
                                         testing_data = testing_feature_data,
                                         number_trees = 105,
                                         number_features = 23)
  print("Baseline data:")
  baseline_results
}
