#| Animal Behaviour Classification using Machine Learning
#| 
#| This script implements a comparative analysis of machine learning approaches
#| for classifying animal behaviors from accelerometer data. It compares:
#| - One-Class Classification (OCC)
#| - Binary Classification
#| - Multi-class Classification


# dataset_name <- "Vehkaoja_Dog"
dataset_name <- "Ladds_Seal"
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
  parallelly,      # Parallel processing utilities
  plotly,          # Interactive plotting
  purrr,           # Functional programming tools
  pROC,            # ROC curve analysis
  rBayesianOptimization,  # Bayesian optimization
  MLmetrics,       # Machine learning metrics
  ranger,          # Fast random forests
  tsfeatures,      # Time series feature extraction
  tidyverse,       # Data manipulation and visualization
  zoo,             # Time series functions
  patchwork        # Combine plots
)

# Global Variables -------------------------------------------------------
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
  "ModelTuningFunctions.R",            # Model optimisation
  "PerformanceCalculationFunctions.R", # Functions for perofmrance and baselines
  "OtherFunctions.R",                  # Utility functions
  "PlotFunctions.R"                    # Generating performance plots
)

invisible(lapply(function_files, function(file) {
  source(file.path(base_path, "Scripts", "Functions", file))
}))

# Analysis Pipeline ----------------------------------------------------

# 1. Split Data into Training and Test Sets
source(file.path(base_path, "Scripts", "SplitTestData.R"))

# 2. Data Exploration and Clustering
source(file.path(base_path, "Scripts", "DataExploration.R"))

# 3. Data Preprocessing, Feature Engineering, and re-clustering
source(file.path(base_path, "Scripts", "Preprocessing.R"))
source(file.path(base_path, "Scripts", "ClusteringBehaviours.R"))

# 4. Hyperparameter Optimization
source(file.path(base_path, "Scripts", "HpoOptimisation.R"))

# 5. Model Evaluation
source(file.path(base_path, "Scripts", "TestBestModels.R"))

# 6. Results Visualization and Comparison
source(file.path(base_path, "Scripts", "PlottingPerformance.R"))
source(file.path(base_path, "Scripts", "PlotPredictions.R"))
