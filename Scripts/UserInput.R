# ---------------------------------------------------------------------------
# Defining Parameters for OCC Model Tuning, Random Additional Atuff
# ---------------------------------------------------------------------------
## Model Selection Options
targetActivity_options <- c("Eating", "Walking")

## Axes to Include in Model (just in case you don't have all of them)
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")

## % of individuals for hold-out test set and k-fold validation set
test_proportion <- 0.2
validation_proportion <- 0.2

## Number of folds for validation
k_folds <- 3

## Method
feature_selection_method <- "RF" 

## features type as in timeseries from tsfeatures package and/or statistical from Tatler et al., 2018
features_type <- c("timeseries", "statistical")