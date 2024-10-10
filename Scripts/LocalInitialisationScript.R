# One Class Classification on Animal Accelerometer Data --------------------

interactive <- FALSE # am I running it or playing with it?

# User Defined Variables ---------------------------------------------------
# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
dataset_name <- "Ladds_Seal"
sample_rate <- 25


# Set up ------------------------------------------------------------------
# load packages
library(pacman)
p_load(
  bench, caret, data.table, e1071, future,
  plotly, PRROC, purrr, pROC, rBayesianOptimization,
  randomForest, tsfeatures, tidyverse, umap, zoo
)
#library(h2o) # for UMAP, but takes a while so ignore unless necessary

# load in the scripts
scripts <-
  list(
    "BaselineSVM.R",
    "DataExploration.R",
    "FeatureGeneration.R",
    "FeatureSelection.R",
    "OtherFunctions.R",
    "ModelTuning.R"
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


# Create hold out test data -----------------------------------------------

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
  # save these
  fwrite(data_test,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  fwrite(data_other,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")
         ))
}


# Explore Data ------------------------------------------------------------

if (interactive == TRUE){
  total_individuals <- length(unique(data_other$ID))
  plot_trace_examples <- plotTraceExamples(behaviours = unique(data_other$Activity), 
                                         data = data_other, 
                                         individuals = 9,
                                         n_samples = 250, 
                                         n_col = 4)

  # regroup the behaviours 
  visualisation_data <- data_other %>% mutate(Activity =
                                              ifelse(Activity %in% c("swimming", "moving", "out", "lying"), Activity, "Other")) %>%
                                     filter(!Activity == "Other")

  plot_activity_by_ID <- plotActivityByID(data = visualisation_data, 
                                        frequency = sample_rate, 
                                        colours = length(unique(data_other$ID)))

  # specify the target behaviours here
  plot_behaviour_durations <- BehaviourDuration(data = data_other, 
                                              sample_rate = sample_rate, 
                                              target_activities = target_activities)
  
  plot <- plot_behaviour_durations$duration_plot
  stats <- plot_behaviour_durations$duration_stats
}

  # from the above, manually specify variables 
  # these may in some cases be different, in which case I need to change this #TODO
  target_activities <- c("swimming", "moving", "still", "chewing")
  window_length <- 1
  overlap_percent <- 0 # also specify this, but I will generally choose 0

# generate all features, unless that has already been done
#(it can be very time consuming so I tend to save it when I've done it)

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

if(interactive == TRUE){
  # Using a UMAP, plot samples of the behaviours to determine which are easy to find
  vis_feature_data <- feature_data[1:10000,] %>%
    select_if( ~ !is.na(.[1])) %>% na.omit() %>%
    mutate(Activity =
             ifelse(Activity %in% c("swimming", "moving", "out", "lying"), Activity, "Other"))
  numeric_features <- vis_feature_data %>% select(-c('Activity', 'Time', 'ID'))
  labels <- vis_feature_data %>% select('Activity')

# UMAP Visualisation ------------------------------------------------------
  UMAP <- UMAPReduction(
    numeric_features,
    labels,
    minimum_distance = 0.01,
    num_neighbours = 5,
    shape_metric = 'manhattan',
    spread = 2
  )
  
  UMAP$UMAP_2D_plot
  UMAP$UMAP_3D_plot
}
  
# this section of the code iterates through hyperparameter combinations for OCC
# PR-ROC for the target class is recorded and saved in the output table

# Tuning model hyperparameters --------------------------------------------
# example data for getting this working
#subset_data <- feature_data %>% group_by(ID, Activity) %>% slice(1:20) %>%ungroup() %>%setDT()

# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.01, 0.1),
  gamma = c(0.01, 0.1),
  number_trees = c(100, 500),
  number_features = c(10, 30)
)

for (beh in target_activities) {
  #beh <- target_activities[2]
  target_activity <- beh
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
    # Number of random initialization points
    init_points = 5,
    # Number of iterations for Bayesian optimization
    n_iter = 15,
    # Acquisition function; can be 'ucb', 'ei', or 'poi'
    acq = "ucb",
    kappa = 2.576      # Trade-off parameter for 'ucb'
  )
}
 
# write out the best performing hyperparameters

# Testing highest performing hyperparmeters -------------------------------

# target_activity <- "Walking"
# number_trees <- 105
# number_features <- 23
# kernel <- "radial"
# nu <- 0.092827
# gamma <- 0.08586
# 
# # make a SVM with training data
# ## load in training data and select features and target data ####
# training_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
# selected_feature_data <- featureSelection(training_data, number_trees, number_features)
# target_selected_feature_data <- selected_feature_data[Activity == as.character(target_activity),!label_columns, with = FALSE]
# ## create the optimal SVM ####
# optimal_single_class_SVM <-
#   do.call(
#     svm,
#     list(
#       target_selected_feature_data,
#       y = NULL,
#       type = 'one-classification',
#       nu = nu,
#       scale = TRUE,
#       kernel = kernel,
#       gamma = gamma
#     )
#   )
# 
# ## load in the test data and generate appropriate features ####
# if (file.exists(file.path(
#   file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")
# )))) {
#   # if this has been run before, just load in
#   testing_feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
# } else { 
#   # calculate and save
#   testing_data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")))
#   testing_feature_data <- generateFeatures(window_length, sample_rate, overlap_percent, data, normalise, features_type)
#   fwrite(data_test,
#          file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
# }
# 
# # calculate performance of the final model in various conditions ####
# training_results <- finalModelPerformance(mode = "training", 
#                                           training_data = target_selected_feature_data, 
#                                           optimal_model = optimal_single_class_SVM)
# 
# testing_results <- finalModelPerformance(mode = "testing", 
#                                           training_data = target_selected_feature_data, 
#                                           optimal_model = optimal_single_class_SVM, 
#                                           testing_data = testing_feature_data, 
#                                           target_activity = target_activity)
# 
# random_results <- finalModelPerformance(mode = "random", 
#                                          training_data = target_selected_feature_data, 
#                                          optimal_model = optimal_single_class_SVM, 
#                                          testing_data = testing_feature_data, 
#                                          target_activity = target_activity)
# 
# baseline_results <- baselineMultiClass(training_data = training_data, 
#                                        testing_data = testing_feature_data, 
#                                        number_trees = 105, 
#                                        number_features = 23)
