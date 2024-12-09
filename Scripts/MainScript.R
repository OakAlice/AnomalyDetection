# One-Class Classification for Animal Accelerometer Behaviour -------------
#| An R script developed for the second chapter of my PhD
#| Intended for comparing the performance of One-class, Binary, and Multi-class ML models

# Set Up ------------------------------------------------------------------

# base_path <- "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
base_path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
# dataset_name <- "Ladds_Seal"
dataset_name <- "Vehkaoja_Dog"

window_settings <- list(Vehkaoja_Dog = list(sample_rate = 100, 
                                    window_length = 1, 
                                    overlap_percent = 50,
                                    target_activities = c("Walking", "Eating", "Shaking", "Lying chest")),
                Ladds_Seal = list(sample_rate = 25,
                                  window_length = 1, 
                                  overlap_percent = 50,
                                  target_activities = c("swimming", "facerub",   "still",    "chewing"))
                        )

sample_rate <- window_settings[[dataset_name]]$sample_rate
window_length <- window_settings[[dataset_name]]$window_length
overlap_percent <- window_settings[[dataset_name]]$overlap_percent
target_activities <- window_settings[[dataset_name]]$target_activities
                        
# install.packages("pacman")
library(pacman)
p_load(
  bench, caret, data.table, e1071, future, future.apply, parallelly,
  plotly, PRROC, purrr, pROC, rBayesianOptimization, MLmetrics, ranger,
  randomForest, tsfeatures, tidyverse, umap, zoo, tinytex, patchwork
)

# global variables
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")
balance <- "stratified_balance"

# Load in the functions ---------------------------------------------------
function_files <- c(
  "FeatureGenerationFunctions.R",
  "FeatureSelectionFunctions.R",
  "ModelTuningFunctions.R",
  "OtherFunctions.R",
  "CalculatePerformanceFunctions.R"
)

invisible(lapply(function_files, function(file) source(file.path(base_path, "Scripts", "Functions", file))))

# Source the scripts that execute stages of the model building ------------

# Split Test Data ---------------------------------------------------------
source(file.path(base_path, "Scripts", "SplitTestData.R"))

# Visualise and Recluster Data --------------------------------------------
source(file.path(base_path, "Scripts", "DataExploration.R"))
source(file.path(base_path, "Scripts", "ClusteringBehaviours.R"))

# Preprocessing and making features ---------------------------------------
source(file.path(base_path, "Scripts", "Preprocessing.R"))

# HPO  --------------------------------------------------------------------
source(file.path(base_path, "Scripts", "HpoOptimisation.R"))

# Test best options -------------------------------------------------------
source(file.path(base_path, "Scripts", "TestBestModels.R"))

# Compare the models ------------------------------------------------------
source(file.path(base_path, "Scripts", "PlottingPerformance.R"))
source(file.path(base_path, "Scripts", "PlotPredictions.R"))
