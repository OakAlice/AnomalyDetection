# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data                  ####
# ---------------------------------------------------------------------------

#### REMEMBER TO ACCOUNT FOR ADDING THE GENERALISED CATEGORIES TO THE DATA ####

# script mode
# mark TRUE what stage you want to execute
# exploration for generating PDF, tuning for HPO finding, testing for final validation
exploration <- FALSE
tuning <- TRUE
testing <- FALSE

# User Defined Variables ---------------------------------------------------
# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
# dataset_name <- "Ladds_Seal"
dataset_name <- "Vehkaoja_Dog"
sample_rate <- 100

# Set up ------------------------------------------------------------------
#library(renv)
# commented out as I already have this
# if (file.exists(file.path(base_path, "renv.lock"))) {
#   renv::restore() # this will install all the right versions of the packages
# }

library(pacman)
p_load(
  bench, caret, data.table, e1071, future, future.apply, parallelly,
  plotly, PRROC, purrr, pROC, rBayesianOptimization,
  randomForest, tsfeatures, tidyverse, umap, zoo, tinytex
)
# note that tinytex needs this too -> tinytex::install_tinytex()
#library(h2o) is for UMAP, but takes a while so ignore unless necessary

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

# some other things I need defined globally :'(
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
# need to check whether this works yet
if (exploration == TRUE) {
  tryCatch({
  # Knit the ExploreData.Rmd file as a PDF and save it to the output folder in base dir
    rmarkdown::render(
      input = file.path(base_path, "Scripts", "ExploreData.Rmd"),
      output_format = "pdf_document",
      output_file = paste0(dataset_name, "_exploration.pdf"),  # Only file name here
      output_dir = file.path(base_path, "Output"),  # Use output_dir for the path
      params = list(
        base_path = base_path,
        dataset_name = dataset_name,
        sample_rate = sample_rate
      )
    )
  message("Exploration PDF saved to: ", output_file)
  }, error = function(e) {
    message("Error in making the data exploration pdf: ", e$message)
    stop()
  })
}

# look at the file and then manually specify details below
#target_activities <- c("swimming", "moving", "still", "chewing")
target_activities <- c("Walking", "Lying chest", "Eating", "Shaking")
window_length <- 1 # TODO: Add that each behaviour has a different one
overlap_percent <- 0

# Adding activity category columns ----------------------------------------
# specifying categories for the behaviours so we can test different combinations

for (condition in c("other", "test")){
  data <- fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
  new_column_data <- renameColumns(data, dataset_name)
  fwrite(new_column_data, file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
}

# Feature Generation ------------------------------------------------------

####### CHANGE THIS TO ACCOUTN FOR MULTIPLE ACTIVITY COLUMNS
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

# Tuning OCC model hyperparameters --------------------------------------
# this section of the code iterates through hyperparameter combinations for OCC
# PR-AUC for the target class is optimised

# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.001, 0.1),
  gamma = c(0.001, 0.1),
  kernel = c(1, 2, 3),
  number_trees = c(100, 500),
  number_features = c(10, 75)
)

ensure.dir(file.path(base_path, "Output"))
if (tuning == TRUE){

  for (target_activity in target_activities) {
    print(target_activity)
    feature_data <- feature_data %>% select(-c("Other", "GeneralisedActivity"))
      
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = function(nu, gamma, kernel, number_trees, number_features) {
        modelTuning(
          feature_data = feature_data,  # Pass feature_data as a fixed argument
          nu = nu,
          kernel = kernel,
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
      },
      bounds = bounds,
      init_points = 5,
      n_iter = 10,
      acq = "ucb",
      kappa = 2.576 
    )
  }
}

# Tuning multiclass model hyperparameters ---------------------------------
# this section of the code tunes the multiclass model, optimising macro average PR-AUC
# the same bounds as in the above section are used for comparison sake

if (tuning == TRUE){
  for (behaviours in c("Activity", "Other", "GeneralisedActivity")){
    
  Activity <- feature_data[Activity == behaviours] # account for this!!!
    
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = function(nu, gamma, kernel, number_trees, number_features) {
        multiclassModelTuning(
          multiclass_data = multiclass_data,
          nu = nu,
          kernel = kernel,
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
      },
      bounds = bounds,
      init_points = 3,
      n_iter = 3,
      acq = "ucb",
      kappa = 2.576 
    )
  }
}

# Testing highest performing hyperparmeters for OCC ----------------------
# currently have to write out the best performing hyperparameters then run
# haven't got it automated yet

target_activity <- "Lying chest"
number_trees <- 100
number_features <- 22.05
kernel <- "radial"
nu <- 0.03
gamma <- 0.03

if (testing == TRUE){
  #### REMEMBER TO ACCOUNT FOR THE MULTIPLE BEHAVIOUR COLUMNS 
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
  testing_feature_data <- testing_feature_data %>% select(-c("Other", "GeneralisedActivity"))
  
  # # make a SVM with training data
  # ## load in training data and select features and target data ####
  training_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
  training_data <- training_data %>% select(-c("Other", "GeneralisedActivity"))
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
  
  # save this model
  model_path <- file.path(base_path, "Output", "Models", paste0(target_activity, "_", dataset_name, "_model.rda"))
  save(optimal_single_class_SVM, file = model_path)
  
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
} 


# Testing highest performing hyperparmeters for multi----------------------
multi <- "generalised" # "Other", "generalised"
number_trees <- 232
number_features <- 45
kernel <- "radial"
nu <- 0.09
gamma <- 0.003

training_feature_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
testing_feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

# classes or not
if (multi == "Other"){
  training_feature_data <- training_feature_data %>% select(-c("Activity", "GeneralisedActivity")) %>%
    rename(Activity = OtherActivity)
  testing_feature_data <- testing_feature_data %>% select(-c("Activity", "GeneralisedActivity")) %>%
    rename(Activity = OtherActivity)
} else if (multi == "generalised"){
  training_feature_data <- training_feature_data %>% select(-c("Activity", "OtherActivity")) %>%
    rename(Activity = GeneralisedActivity)
  testing_feature_data <- testing_feature_data %>% select(-c("Activity", "OtherActivity")) %>%
    rename(Activity = "GeneralisedActivity")
} else if (multi == "all"){
  training_feature_data <- training_feature_data %>% select(-c("GeneralisedActivity", "OtherActivity"))
  testing_feature_data <- testing_feature_data %>% select(-c("GeneralisedActivity", "OtherActivity"))
}

baseline_results <- baselineMultiClass(training_data = training_feature_data,
                                         testing_data = testing_feature_data,
                                         number_trees = number_trees,
                                         number_features = number_features,
                                         kernel = kernel,
                                         cost = nu,
                                         gamma = gamma)
print(paste0("Baseline ", multi, ":"))
baseline_results
