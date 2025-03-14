#| Animal Behaviour Classification using Machine Learning
#| 
#| This program implements a comparative analysis of machine learning approaches
#| for classifying animal behaviours from accelerometer data.
#| It compares SVM and Tree based systems for
#| - One-Class Classification (OCC)
#| - Binary Classification
#| - Multi-class Classification

# dataset_name <- "Vehkaoja_Dog"
dataset_name <- "Anguita_Human"
base_path <- "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
# base_path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"

# Set Up ------------------------------------------------------------------
# User defined variables for each dataset
#' @param sample_rate Sampling frequency in Hz
#' @param window_length Window size in seconds
#' @param overlap_percent Overlap between windows as a whole number
#' @param target_activities Behaviours to classify in the OCC and Binary models
window_settings <- list(
  Vehkaoja_Dog = list(
    sample_rate = 100,
    window_length = 1,
    overlap_percent = 50,
    target_activities = c("Walking", "Eating", "Shaking", "Lying chest")
  ),
  Ladds_Seal = list(
    sample_rate = 25,
    window_length = 1,
    overlap_percent = 50,
    target_activities = c("swimming", "still", "chewing", "facerub")
  ),
  Anguita_Human = list(
    sample_rate = "not sure",
    window_length = "not sure",
    overlap_percent = "not sure",
    target_activities = c("WALKING", "SITTING", "STANDING")
  )
)

sample_rate <- window_settings[[dataset_name]]$sample_rate
window_length <- window_settings[[dataset_name]]$window_length
overlap_percent <- window_settings[[dataset_name]]$overlap_percent
target_activities <- window_settings[[dataset_name]]$target_activities

# Load Required Packages --------------------------------------------------
library(pacman)
p_load(
  caret,           # Classification and regression training
  data.table,      # Fast data manipulation
  e1071,           # SVM implementation
  future,          # Parallel processing
  future.apply,    # Parallel apply functions
  isotree,         # Anomaly detection forest package
  parallelly,      # Parallel processing utilities
  plotly,          # Interactive plotting
  purrr,           # Functional programming tools
  pROC,            # ROC curve analysis
  rBayesianOptimization,  # Bayesian optimization
  MLmetrics,       # Machine learning metrics
  ranger,          # Fast random forests
  randomForest,    # normal random forest
  reshape2,        # melting
  rpart,           # decision tree
  tree,            # decision tree 
  tsfeatures,      # Time series feature extraction
  tidyverse,       # Data manipulation and visualization
  zoo,             # Time series functions
  patchwork        # Combine plots
)

# Global Variables -------------------------------------------------------
ML_method <- "SVM" # or "Tree"
training_sets <- c("all", "some", "target") # this is the behaviours that appear in training set
training_set <- c("all")
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")
balance <- "stratified_balance"

# Custom Functions -------------------------------------------------------
function_files <- c(
  "FeatureGenerationFunctions.R",      # Feature extraction
  "FeatureSelectionFunctions.R",       # Feature selection methods
  "SVMModelTuningFunctions.R",         # SVM Model optimisation
  "TreeModelTuningFunctions.R",        # Tree Model optimisation
  "PerformanceCalculationFunctions.R", # Functions for performance and baselines
  "OtherFunctions.R",                  # Utility functions
  "PlotFunctions.R"                    # Generating performance plots
)

invisible(lapply(function_files, function(file) {
  source(file.path(base_path, "Scripts", "Functions", file))
}))

# Analysis Pipeline ----------------------------------------------------

# Split Data into Training and Test Sets
source(file.path(base_path, "Scripts", "SplitTestData.R"))

# Data Exploration and Clustering
source(file.path(base_path, "Scripts", "DataExploration.R"))

# Data Preprocessing, Feature Engineering, and re-clustering
source(file.path(base_path, "Scripts", "Preprocessing.R"))
source(file.path(base_path, "Scripts", "ClusteringBehaviours.R"))

# Remove different behaviours from the training set 
source(file.path(base_path, "Scripts", "RemoveTrainingInformation.R"))


# Hyperparameter Optimization
# options for SVM and Tree based comparisons
source(file.path(base_path, "Scripts", "SVMHpoOptimisation.R"))
source(file.path(base_path, "Scripts", "TreeHpoOptimisation.R"))

# Generate optimal models
source(file.path(base_path, "Scripts", "SVMTrainBestModels.R"))
source(file.path(base_path, "Scripts", "TreeTrainBestModels.R"))

# Model Evaluation
source(file.path(base_path, "Scripts", "SVMTestBestModels.R"))
source(file.path(base_path, "Scripts", "TreeTestBestModels.R"))

# Results Visualization and Comparison
source(file.path(base_path, "Scripts", "PlottingPerformance.R"))
source(file.path(base_path, "Scripts", "PlotPredictions.R"))
