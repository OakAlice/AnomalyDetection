# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data                  ####
# ---------------------------------------------------------------------------

# script mode
# mark TRUE what stage you want to execute
# exploration for generating PDF, tuning for HPO finding, testing for final validation
exploration   <- FALSE
renamed       <- TRUE # whether I have already added in the general beh categories
tuningOCC     <- FALSE
tuningMulti   <- FALSE
testingOCC    <- FALSE
testingMulti  <- TRUE

# User Defined Variables ---------------------------------------------------
# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
dataset_name <- "Ladds_Seal"
#dataset_name <- "Vehkaoja_Dog"
sample_rate <- 25

# install.packages("pacman")
library(pacman)
p_load(
  bench, caret, data.table, e1071, future, future.apply, parallelly,
  plotly, PRROC, purrr, pROC, rBayesianOptimization,
  randomForest, tsfeatures, tidyverse, umap, zoo, tinytex
)
# note that tinytex needs this too -> tinytex::install_tinytex()
#library(h2o) is for UMAP, but takes a while so ignore unless necessary

# some other things I need defined globally :'(
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")

# load in the function scripts
function_scripts <-
  list(
    "Scripts/Functions/BaselineSVMFunctions.R",
    "Scripts/Functions/FeatureGenerationFunctions.R",
    "Scripts/Functions/FeatureSelectionFunctions.R",
    "Scripts/Functions/OtherFunctions.R",
    "Scripts/Functions/ModelTuningFunctions.R",
    "Scripts/Functions/CalculatePerformanceFunctions.R"
  )

# Function to source scripts and handle errors
successful <- TRUE
source_script <- function(script) {
  tryCatch(
    source(file.path(base_path, script)),
    error = function(e) {
      successful <<- FALSE
      message(paste("Error sourcing script:", script))
    }
  )
}
walk(function_scripts, source_script)

# Split Test Data ---------------------------------------------------------
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


# Data Exploration --------------------------------------------------------
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

# Specify Variables -------------------------------------------------------
# look at the pdf file and then specify details below
target_activities <- c("swimming", "moving", "still", "chewing")
target_activity <- "swimming"
#target_activities <- c("Walking", "Lying chest", "Eating", "Shaking")
window_length <- 1 # TODO: Add that each behaviour has a different one
overlap_percent <- 0


# Adding activity category columns ----------------------------------------
# specifying categories for the behaviours so we can test different combinations
if (renamed == FALSE){
  for (condition in c("other", "test")){
    data <- fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
    new_column_data <- renameColumns(data, dataset_name)
    fwrite(new_column_data, file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
  }
}


# Feature Generation ------------------------------------------------------
if (file.exists( file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))) {
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
} else {
  
  types <- c("test", "other")
  for (type in types){
    
    data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_", type, ".csv")))
    
    for (id in unique(data$ID))  {
      dat <- data %>% filter(ID == id)
      
      feature_data <-
        generateFeatures(
          window_length,
          sample_rate,
          overlap_percent,
          raw_data = dat,
          features = features_type
        )
      # save it 
      fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_", type, "_features.csv")))
    }
    
    # stitch all the id feature data back together
    files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
    pattern <- paste0(dataset_name, ".*", type)
    matching_files <- grep(pattern, files, value = TRUE)
    
    feature_data_list <- lapply(matching_files, read.csv)
    feature_data <- do.call(rbind, feature_data_list)
    
    # save this as well
    fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", type, "_features.csv")))
  }
}


# Tuning OCC model hyperparameters --------------------------------------
# PR-AUC for the target class is optimised
target_activities <- c("chewing", "swimming", "scratch", "still")


# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.001, 0.1),
  gamma = c(0.001, 0.1),
  kernel = c(1, 2, 3),
  number_trees = c(100, 500),
  number_features = c(10, 75)
)

feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
target_activities <- c("swimming", "scratch", "still", "chewing")
feature_data <- feature_data %>% select(-c("OtherActivity", "GeneralisedActivity")) %>%
  as.data.table()

ensure.dir(file.path(base_path, "Output"))

#if (tuningOCC == TRUE){

  for (target_activity in target_activities) {
    print(target_activity)
      
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
#}

# Tuning multiclass model hyperparameters ---------------------------------
# this section of the code tunes the multiclass model, optimising macro average PR-AUC
# the same bounds as in the above section are used for comparison sake
# remmeber to account for there being multiple types of activity columns 
behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")

feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
feature_data <- feature_data %>% as.data.table()


if (tuningMulti == TRUE){
  for (behaviours in behaviour_columns){
    
    behaviours <- "GeneralisedActivity"
    
    
    multiclass_data <- feature_data %>%
      select(-(setdiff(behaviour_columns, behaviours))) %>%
      rename("Activity" = !!sym(behaviours)) %>%
      select(-"...6") # remove this random column
    
    if(behaviours == "GeneralisedActivity"){
      multiclass_data <- multiclass_data %>% filter(!Activity == "")
    }
    
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
      n_iter = 5,
      acq = "ucb",
      kappa = 2.576 
    )
  }
}
