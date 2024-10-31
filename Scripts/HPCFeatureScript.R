#!/bin/bash/Rscript

#---------------------------------------------------------------------------
# HPC Script                                                            ####
#---------------------------------------------------------------------------
# this script is different because it has no interactivity
# it is expected that you would have determined the behaviours and window lengths
# currently it just does the feature generation

#---------------------------------------------------------------------------
# Set Up                                                                 ####
#---------------------------------------------------------------------------

# information from the HPC command
args <- commandArgs(trailingOnly = TRUE)
# args <- c("C:/Users/oaw001/Documents/AnomalyDetection", "package/path", "Pagona_Bear", c("walk", "rest", "eat"), 10, 16)
base_path <- args[1]
package_path <- args[2]
dataset_name <- args[3]
target_activities <- args[4]
window_length <- args[5]
sample_rate <- args[6]
overlap_percent <- 0 # could change this to user specified, but I will generally choose 0

# load packages
tryCatch({
  library(caret, lib.loc = package_path)
  library(data.table, lib.loc = package_path)
  library(e1071, lib.loc = package_path)
  library(future, lib.loc = package_path)
  library(purrr, lib.loc = package_path)
  library(tsfeatures, lib.loc = package_path)
  library(tidyverse, lib.loc = package_path)
  library(zoo, lib.loc = package_path)
  }, error = function(e) {
    message("Error in loading packages: ", e$message)
})
  
# load in the scripts
scripts <-
  list(
    "FeatureGeneration.R",
    "FeatureSelection.R"
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

# some other things I need defined
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2 
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")

#---------------------------------------------------------------------------
# Create data splits for this run                                       ####
#---------------------------------------------------------------------------

# load in data
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
  # create the save directory
  ensure.dir(file.path(base_path, "Data/Hold_out_test"))
  # save these
  fwrite(data_test,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  fwrite(data_other,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")
         ))
}

#---------------------------------------------------------------------------
# Feature Generation                                                    ####
#---------------------------------------------------------------------------
# generate all features, unless that has already been done

if (file.exists(
  file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")
  ))) {
  feature_data <-
    fread(
      file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")
      ))
} else {
  
  for (id in unique(data_other$ID))  {
    
    # breaking it up by individual reduces chance that error will blow whole system
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
    ensure.dir(file.path(base_path, "Data/Feature_data"))
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
