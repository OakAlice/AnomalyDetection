
# WildOne: One Class Classification for Animal Accelerometer Behav --------


# Set Up ------------------------------------------------------------------

#base_path <- "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection"
base_path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
dataset_name <- "Ladds_Seal"
#dataset_name <- "Vehkaoja_Dog"
sample_rate <- 25
window_settings <- list(Vehkaoja_Dog = list(window_length = 1, overlap_percent = 50),
                        Ladds_Seal = list(window_length = 1, overlap_percent = 50))

target_activities <- c("swimming", "facerub",   "still",    "chewing")

# install.packages("pacman")
library(pacman)
p_load(
  bench, caret, data.table, e1071, future, future.apply, parallelly,
  plotly, PRROC, purrr, pROC, rBayesianOptimization,
  randomForest, tsfeatures, tidyverse, umap, zoo, tinytex
)

# global variables
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")

# Load in the functions ---------------------------------------------------
function_files <- c(
  "FeatureGenerationFunctions.R",
  "FeatureSelectionFunctions.R",
  "ModelTuningFunctions.R",
  "OtherFunctions.R",
  "CalculatePerformanceFunctions.R",
  "BaselineSVMFunctions.R"
)

invisible(lapply(function_files, function(file) source(file.path(base_path, "Scripts", "Functions", file))))

# Split Test Data ---------------------------------------------------------
source(file.path(base_path, "Scripts", "SplitTestData.R"))

# Visualise Data ----------------------------------------------------------
source(file.path(base_path, "Scripts", "DataExploration.R"))

# Reclustering the behaviours for multi-class models ----------------------
source(file.path(base_path, "Scripts", "ClusteringBehaviours.R"))

# Preprocessing and making features ---------------------------------------
source(file.path(base_path, "Scripts", "Preprocessing.R"))

# HPO  --------------------------------------------------------------------
source(file.path(base_path, "Scripts", "HpoOptimisation.R"))

# Test best options -------------------------------------------------------
source(file.path(base_path, "Scripts", "TestBestModels.R"))

