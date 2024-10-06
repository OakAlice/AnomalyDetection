# ---------------------------------------------------------------------------
# Defining Parameters for OCC Model Tuning
# ---------------------------------------------------------------------------
## Model Selection Options
# Choose the type of model to be used in the classification task.
# Current options: SVM (other options such as Autoencoder, GMM, PPNN, KNN to be added later)
model_options <- "SVM"

# move this to be in the dictionaries
targetActivity_options <- c("Eating", "Walking")
  
## Axes to Include in Model (just in case you don't have all of them)
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

# ---------------------------------------------------------------------------
# Validation Parameters 
# ---------------------------------------------------------------------------

## % of individuals for hold-out test set and k-fold validation set
test_proportion <- 0.2
validation_proportion <- 0.2

## Number of folds for validation
k_folds <- 3

# ---------------------------------------------------------------------------
# Feature Selection Parameters
# ---------------------------------------------------------------------------

## Method
# Current support for RF and UMAP.
feature_selection_method <- "RF" 

## Feature Normalization
# Current Options: "Standardisation", "MinMaxScaling"
feature_normalisation_options <- c("z-scale")

# features type as in timeseries from tsfeatures package and/or statistical from Tatler et al., 2018
features_type <- c("timeseries", "statistical")
number_features <- 50



