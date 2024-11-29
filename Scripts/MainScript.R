
# WildOne: One Class Classification for Animal Accelerometer Behav --------


# Set Up ------------------------------------------------------------------

base_path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
dataset_name <- "Ladds_Seal"
#dataset_name <- "Vehkaoja_Dog"
sample_rate <- 100

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


# Split Test Data ---------------------------------------------------------


