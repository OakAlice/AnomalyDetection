# Feature selection and dimensionality reduction


# Feature Selection for Activity Classification
# This function performs feature selection on training data for activity classification.
# It applies several preprocessing steps and uses Random Forest for feature importance ranking.
#
#' @param training_data A data frame containing the training data with features and activity labels
#' @param number_features Integer specifying the desired number of features to select
#' @param corr_threshold Numeric threshold (0-1) for removing highly correlated features (default: 0.9)
#'
#' @return A data.table containing the selected features and Activity column, or NULL if an error occurs
#
# @details
# The function performs the following:
# 1. Removes non-numeric columns (Activity, Time, ID)
# 2. Filters out columns with >50% NA values
# 3. Removes zero-variance features
# 4. Removes highly correlated features based on correlation threshold
# 5. Uses Random Forest or PCA to select top features
featureSelection <- function(model, training_data, number_features, corr_threshold = 0.9) {
  tryCatch(
    {
      # Ensure this is a whole number
      number_features <- round(number_features, 0)
      
      clean_columns <- cleanTrainingData(training_data, corr_threshold)
      training_data_clean <- training_data[, ..clean_columns]
      training_data_clean <- training_data_clean[complete.cases(training_data_clean), ]
      
      # Check for multiple classes in Activity column
      if (length(unique(training_data_clean$Activity)) <= 1) {
        message("Only one class detected; implementing PCA feature selection.")
        flush.console()

        # PCA feature selection finds variables that best describe target class
        top_features <- pca_feature_selection(
          data = training_data_clean,
          number_features = number_features,
          model = "OCC"
        )

      } else {
        # Random Forest feature selection
        if (ncol(training_data_clean) > number_features) {
          message("Starting Random Forest feature selection.")
          flush.console()

          # Sample 75% of data for feature selection
          sampled_data <- training_data_clean %>%
            group_by(Activity) %>%
            slice_sample(prop = 0.75) %>%
            as.data.frame()

          top_features <- featureSelectionRF(
            data = sampled_data,
            n_trees = 500,
            number_features = as.numeric(number_features)
          )

          if (is.null(top_features)) {
            message("Random Forest feature selection failed; returning all features.")
            flush.console()
            top_features <- names(training_data_clean)
          }
        } else {
          top_features <- names(training_data_clean)
        }
      }

      # Include "Activity" in the final feature set
      top_features <- c(top_features, "Activity")

      # Return the final dataset with selected features
      return(top_features)
    },
    error = function(e) {
      message("Error in featureSelection: ", e$message)
      return(NULL)
    }
  )
}

# Random Forest Feature Selection
# 
# Performs feature selection using Random Forest importance scores.
# 
# @param data A data frame containing the features and Activity column
# @param n_trees Number of trees to grow in the Random Forest model
# @param number_features Maximum number of top features to select
# 
# @return A character vector containing the names of the selected top features
# @details
# This function fits a Random Forest model using the ranger package and ranks features
# based on their impurity importance scores. It returns the top N most important features,
# where N is specified by number_features.
# 
# The function includes error handling and will return NULL if the Random Forest model fails.
# The Activity column is automatically converted to a factor.

featureSelectionRF <- function(data, n_trees, number_features) {
  tryCatch(
    {
      data <- data %>%
        mutate(Activity = factor(Activity))

      # Fit Random Forest model with ranger
      rf_model <- ranger(
        dependent.variable.name = "Activity",
        data = data,
        num.trees = n_trees,
        importance = "impurity", # Equivalent to MeanDecreaseAccuracy
        classification = TRUE,
        probability = FALSE
      )

      # Extract and rank feature importance
      importance_df <- as.data.frame(rf_model$variable.importance) %>%
        tibble::rownames_to_column("Feature") %>%
        dplyr::arrange(desc(rf_model$variable.importance)) %>%
        dplyr::slice_head(n = min(number_features, nrow(.)))

      top_features <- importance_df$Feature

      return(top_features)
    },
    error = function(e) {
      message("Error in Random Forest feature selection: ", e$message)
      return(NULL)
    }
  )
}


# Advanced feature selection using PCA
# @param data Data.table containing feature data
# @param number_features Number of features to select
# @param model Type of model ("OCC", "Binary", or "Multi")
# @param variance_explained Minimum cumulative variance to explain (default 0.95)
# @return Data.table with selected features
pca_feature_selection <- function(data, number_features, model, variance_explained = 0.95) {
  # Remove non-numeric columns but keep Activity for later
  activity_col <- data$Activity
  numeric_data <- data[, !c("Activity"), with = FALSE]
  
  # Scale the data
  scaled_data <- scale(numeric_data)
  
  # Perform PCA
  pca_result <- prcomp(scaled_data)
  
  # Calculate variance explained
  var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
  cum_var_explained <- cumsum(var_explained)
  
  # Determine number of components to keep based on variance explained
  n_components <- min(which(cum_var_explained >= variance_explained))
  
  # Get loadings for the selected components
  loadings <- abs(pca_result$rotation[, 1:n_components])
  
  # Calculate feature importance differently based on model type
  feature_importance <- if(model == "OCC") {
    # For OCC: Focus on magnitude of contribution to main components
    rowSums(loadings)
  } else {
    # For Binary/Multi: Weight components by their explained variance
    weighted_loadings <- sweep(loadings, 2, var_explained[1:n_components], "*")
    rowSums(weighted_loadings)
  }
  
  # Select top features
  top_features <- names(sort(feature_importance, decreasing = TRUE)[1:number_features])
  
  return(top_features)
}


cleanTrainingData <- function(training_data, corr_threshold) {
  # Step 1: Filter numeric features
  numeric_columns <- training_data %>%
    select(-Activity, -Time, -ID) %>%
    mutate(across(everything(), as.numeric))
  
  # Step 2: Remove columns with >50% NA
  valid_cols <- names(numeric_columns)[colMeans(is.na(numeric_columns)) <= 0.5]
  numeric_columns <- numeric_columns[, ..valid_cols]
  
  # Step 3: Remove zero-variance columns
  numeric_columns <- numeric_columns %>% 
    select(where(~ {
      sd_val <- sd(., na.rm = TRUE)
      !is.na(sd_val) && sd_val > 0
    }))
  
  numeric_columns <- numeric_columns[complete.cases(numeric_columns), ]
  
  # Step 5: Remove highly correlated features
  if (ncol(numeric_columns) > 1) {
    # Use Pearson correlation with pairwise complete observations to handle NaNs
    corr_matrix <- cor(numeric_columns, use = "pairwise.complete.obs", method = "pearson")
    
    # Replace NaN with 0 to avoid errors in findCorrelation
    corr_matrix[is.na(corr_matrix)] <- 0
    
    high_corr <- findCorrelation(corr_matrix, cutoff = corr_threshold, names = TRUE)
    numeric_columns <- numeric_columns[, setdiff(names(numeric_columns), high_corr), with = FALSE]
  }
  
  # Check if any features remain
  if (ncol(numeric_columns) == 0) stop("No valid features remaining after preprocessing.")
  
  clean_columns <- c(colnames(numeric_columns), "Activity")
  
  return(clean_columns)
}
