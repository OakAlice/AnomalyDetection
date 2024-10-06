#---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data                  ####
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Set Up                                                                 ####
#---------------------------------------------------------------------------

# load packages
library(pacman)
p_load(data.table, tidyverse, purrr, future.apply, e1071, zoo, caret,
       tsfeatures, umap, plotly, randomForest, pROC, bench, PRROC, rBayesianOptimization)
#library(h2o) # for UMAP, but takes a while so ignore unless necessary

# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
#setwd("C:/Users/oaw001/Documents/AnomalyDetection")

# load in the scripts
scripts <- list("Dictionaries.R", "PlotFunctions.R", "FeatureGeneration.R",
                "FeatureSelection.R", "OtherFunctions.R",
                "UserInput.R", "ModelTuning.R") #, "DataExploration.R")

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

#---------------------------------------------------------------------------
# Define parameters and create data splits for this run                 ####
#---------------------------------------------------------------------------
dataset_name <- "Vehkaoja_Dog"
list_name <- all_dictionaries[[dataset_name]]
movement_data <- get(list_name)

# load in data
move_data <- fread(file.path(base_path, "Data", paste0(movement_data$name, "_Corrected.csv")))

# Split Data ####
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

# explore data in PreProcessingDecisions.R to determine window length and behavioural clustering
# haven't automated this yet
# when complete, add to dictionary

#---------------------------------------------------------------------------
# Feature Generation                                                    ####
#---------------------------------------------------------------------------
# generate all features, unless that has already been done 
#(it can be very time consuming so I tend to save it when I've done it)

if (file.exists(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))){
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
} else {
  feature_data <- generate_features(movement_data, data = data_other, 
                                    normalise = "z_scale", features = features_type)
}

#---------------------------------------------------------------------------
# Choose behaviours to identify                                         ####
#---------------------------------------------------------------------------
# Using a UMAP, plot samples of the behaviours to determine which are easy to find
vis_feature_data <- feature_data[1:2000, ] %>%
  select_if(~ !is.na(.[1])) %>% na.omit()
numeric_features <- vis_feature_data %>% select(-c('Activity', 'Time', 'ID'))
labels <- vis_feature_data %>% select('Activity')
UMAP <- UMAP_reduction(numeric_features, 
                      labels, 
                      minimum_distance = 0.2, 
                      num_neighbours = 10, 
                      shape_metric = 'euclidean')
  

UMAP$UMAP_2D_plot
UMAP$UMAP_3D_plot

#---------------------------------------------------------------------------
# Tuning model hyperparameters                                           ####
#---------------------------------------------------------------------------
# this section of the code iterates through hyperparameter combinations for OCC
# PR-ROC for the target class is recorded and saved in the output table

# example data for getting this working
#subset_data <- feature_data %>% group_by(ID, Activity) %>% slice(1:20) %>%ungroup() %>%setDT()

targetActivity <- "Lying chest"
# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.01, 0.1),  
  kernel = c(0, 1),     
  gamma = c(0.01, 0.1),     
  number_trees = c(100, 500), 
  number_features = c(20, 50) 
)

behList <- c("Tugging", "Sniffing", "Panting")

for (beh in behList){
  
  targetActivity <- beh
# Run the optimization
  bench_time({
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = model_train_and_validate,
      bounds = bounds,
      init_points = 5,    # Number of random initialization points
      n_iter = 15,        # Number of iterations for Bayesian optimization
      acq = "ucb",        # Acquisition function; can be 'ucb', 'ei', or 'poi'
      kappa = 2.576       # Trade-off parameter for 'ucb'
    )
  })

}

#---------------------------------------------------------------------------
# Testing highest performing model hyperparameters                       ####
#---------------------------------------------------------------------------
# write out the best performing hyperparameters

targetActivity <- "Walking"
number_trees <- 105
number_features <- 23
kernel <- "radial"
nu <- 0.092827
gamma <- 0.08586


# load in the test data and generate features
#testing_data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_Labelled_test.csv")))
#testing_data_sample <- testing_data %>% group_by(ID, Activity) %>% slice(1:100) %>% ungroup() %>% setDT()

#testing_feature_data <- generate_features(movement_data, data = testing_data, 
#                                          normalise = "z_scale", features_type = features_type)
# save this for later
#fwrite(testing_feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

# load in training data and select features and target data
training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
selected_feature_data <- feature_selection(training_data, number_trees, number_features)
target_selected_feature_data <- selected_feature_data[Activity == as.character(targetActivity),
                                                   !label_columns, with = FALSE] 
# create the optimal SVM
optimal_single_class_SVM <- do.call(svm, list(target_selected_feature_data, y = NULL, type = 'one-classification', 
                                                nu = nu, scale = TRUE, 
                                                kernel = kernel, 
                                                gamma = gamma))

# calculate performance on the training set ####
decision_scores <- predict(optimal_single_class_SVM, newdata = target_selected_feature_data, decision.values = TRUE)
scores <- as.numeric(attr(decision_scores, "decision.values"))
true_labels <- rep(1, nrow(target_selected_feature_data)) # because they are all from the training class

training_results <- get_performance(scores, ground_truth_labels = true_labels)

# Calculate performance on the test set ####
# Format the test data
top_features <- colnames(target_selected_feature_data)
testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
filtered_test_data <- testing_data[, .SD, .SDcols = c("Activity", top_features)]
balanced_test_data <- filtered_test_data[complete.cases(filtered_test_data), ]

# Balance the testing data
activity_count <- balanced_test_data[Activity == targetActivity, .N]
balanced_test_data[Activity != targetActivity, Activity := "Other"]
balanced_test_data <- balanced_test_data[, .SD[1:activity_count], by = Activity]

# Extract the ground truth labels from the balanced test data
test_labels <- balanced_test_data[, "Activity"]
test_labels <- ifelse(test_labels$Activity == as.character(targetActivity), 1, -1)

# Predict using the trained SVM model
numeric_test_features <- balanced_test_data[, !"Activity"]
decision_scores <- predict(optimal_single_class_SVM, newdata = numeric_test_features, decision.values = TRUE)
predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))

# Get performance for the balanced test data
balanced_test_results <- get_performance(predicted_scores, test_labels)


# Calculate performance on the randomized test set ####
# Randomize 'Activity' column to generate a randomized test set
randomized_test_data <- filtered_test_data[complete.cases(filtered_test_data), ]
randomized_test_data$Activity <- sample(randomized_test_data$Activity)

# Balance the randomized test data
random_activity_count <- randomized_test_data[Activity == targetActivity, .N]
randomized_test_data[Activity != targetActivity, Activity := "Other"]
balanced_random_test_data <- randomized_test_data[, .SD[1:random_activity_count], by = Activity]

# Extract the ground truth labels from the randomized test data
random_test_labels <- balanced_random_test_data[, "Activity"]
random_test_labels <- ifelse(random_test_labels$Activity == as.character(targetActivity), 1, -1)

# Predict using the trained SVM model
numeric_random_test_features <- balanced_random_test_data[, !"Activity"]
random_decision_scores <- predict(optimal_single_class_SVM, newdata = numeric_random_test_features, decision.values = TRUE)
random_predicted_scores <- as.numeric(attr(random_decision_scores, "decision.values"))

# Get performance for the randomized test data
random_test_results <- get_performance(random_predicted_scores, random_test_labels)
