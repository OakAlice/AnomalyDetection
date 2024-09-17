#!/usr/lib/R/bin/Rscript

# ---------------------------------------------------------------------------
# Initialisation Script for Running on the PBS HPC
# ---------------------------------------------------------------------------

# recieving arguments from the run script
args <- commandArgs(trailingOnly = TRUE)
dataset_name <- args[1]
base_path <- args[2]
package_path <- args[3]

# ---------------------------------------------------------------------------
# Set Up, packages and scripts
# ---------------------------------------------------------------------------

# load packages, specifiying where in the HPC they are saved 
package_list <- c("dplyr", "randomForest", "caret", "e1071", "WaveletComp", 
                  "purrr", "cowplot", "scales", "crqa", "pracma", "doParallel", "foreach")

for (package in package_list){
  library(package, lib.loc=package_path)
}

# load in the scripts
scripts <- list("Dictionaries.R", "PlotFunctions.R", "FeatureGeneration.R",
                "FeatureSelection.R", "OtherFunctions.R",
                "UserInput.R", "ModelTuning.R") #, "DataExploration.R")

# Function to source scripts and handle errors
source_script <- function(script_path) {
  if (file.exists(script_path)) {
    source(script_path)
  } else {
    message("Script not found: ", script_path)
  }
}

# Source each script, print if it didn't work
for (script in scripts) {
  script_path <- file.path(base_path, "Scripts", script)
  source_script(script_path)
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

list_name <- all_dictionaries[[dataset_name]]
movement_data <- get(list_name)

# try to read in the data
tryCatch({
  move_data <- fread(file.path(base_path, "Data", paste0(movement_data$name, ".csv")))
}, error = function(e) {
  print(paste("Error reading CSV file:", e))
})

# ---------------------------------------------------------------------------
# Split out test data
# ---------------------------------------------------------------------------

# Split Data ####
tryCatch({
  if (file.exists(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_Labelled_test.csv")))){
    # if this has been run before, just load in the split data  
    data_test <- fread(file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
    data_other <-fread(file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))
  } else {
    # if this is the first time running code for this dataset, create hold-out test set
    unique_ids <- unique(move_data$ID)
    test_ids <- sample(unique_ids, ceiling(length(unique_ids) * test_proportion))
    data_test <- move_data[ID %in% test_ids]
    data_other <- move_data[!ID %in% test_ids]
    # save these
    fwrite(data_test, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
    fwrite(data_other, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))
  }
}, error = function(e) {
  print(paste("Error splitting test data:", e))
})

# explore data in PreProcessingDecisions.R to determine window length and behavioural clustering
# haven't automated this yet
# when complete, add to dictionary

# ---------------------------------------------------------------------------
# Feature Generation and Elimination
# ---------------------------------------------------------------------------
# generate all features, unless that has already been done 
#(it can be very time consuming so I save it when I've done it once)
tryCatch({
  if (file.exists(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))){
    feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
  } else {
    feature_data <- generate_features(movement_data, data = data_other, 
                                      normalise = "z_scale", features = features_type)
    fwrite(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
  }
}, error = function(e) {
  print(paste("Error generating feature data:", e))
})

# ---------------------------------------------------------------------------
# Tuning model hyperparameters
# ---------------------------------------------------------------------------
# this section of the code iterates through hyperparameter combinations for OCC
# AUC for the target class is recorded and saved in the output table

# create all the options # options_df exists in a different script 'OtherFunctions.R'
tryCatch({
  options_df <- expand_all_options(model_hyperparameters_list, feature_hyperparameters_list,
                                 targetActivity_options, model_options, 
                                 feature_selection, feature_normalisation_options, 
                                 nu_options, kernel_options, degree_options)

}, error = function(e) {
  print(paste("Error making combinations:", e))
})

# example data for getting this working
#subset_data <- feature_data %>% group_by(ID, Activity) %>% slice(1:20) %>%ungroup() %>%setDT()

subset_data <- feature_data

# tune the model design by trialing each line in the extended_options_df2
tryCatch({
  model_outcomes <- map_dfr(1:nrow(options_df), ~process_row(options_df[., ], 
                                                             k_folds, 
                                                             subset_data, 
                                                             validation_proportion, 
                                                             feature_selection, 
                                                             base_path, 
                                                             dataset_name, 
                                                             number_features))
}, error = function(e) {
  print(paste("Error tuning models:", e))
})

# ---------------------------------------------------------------------------
# Saving output
# ---------------------------------------------------------------------------

# save to HPC
model_outcomes <- setDT(model_outcomes)
ensure.dir(file.path(base_path, "Output", dataset_name))
fwrite(model_outcomes, file.path(base_path, "Output", dataset_name, paste0(dataset_name, "_model_outcomes_test1.csv")))

