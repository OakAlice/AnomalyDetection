# ---------------------------------------------------------------------------
# Defining Parameters for OCC Model Tuning
# ---------------------------------------------------------------------------
## Model Selection Options
# Choose the type of model to be used in the classification task.
# Current options: SVM (other options such as Autoencoder, GMM, PPNN, KNN to be added later)
model_options <- "SVM"

# move this to be in the dictionaries
targetActivity_options <- c("Eating", "Walking", "Sitting")
  
## Axes to Include in Model (just in case you don't have all of them)
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

# ---------------------------------------------------------------------------
# Tunable Model Hyperparameters
# ---------------------------------------------------------------------------

# Below are SVM-specific hyperparameters - stuff for other methods to be added later
nu_options <- c(0.01, 0.05)
kernel_options <- c("radial", "linear") #"polynomial", "sigmoid
gamma_options <- c(0.001, 0.01)
degree_options <- c(3, 4) # Only if using a polynomial kernel

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
feature_selection <- "RF" 

## Feature Normalization
# Current Options: "Standardisation", "MinMaxScaling"
feature_normalisation_options <- c("z-scale")

# features type as in timeseries from tsfeatures package and/or statistical from Tatler et al., 2018
features_type <- c("timeseries", "statistical")
number_features <- 50

## UMAP hyperparameters 
minimum_distance_options <- c(0.7)
num_neighbours_options <- 10
shape_metric_options <- 'manhattan'  # Other options: 'euclidean', etc.

## Random Forest hyperparameters
number_trees_options <- c(100)
number_features_options <- c(25, 50)



