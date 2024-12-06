# Feature selection and dimensionality reduction

featureSelection <- function(training_data, number_trees, number_features) {
  tryCatch({
    # Step 1: Eliminate features with no variance and high correlation
    potential_features <- tryCatch({
      removeBadFeatures(training_data, threshold = 0.9)  # Remove features with no variance or high correlation
    }, error = function(e) {
      message("Error in removeBadFeatures: ", e$message)
      return(NULL)
    })
    
    if (is.null(potential_features)) {
      return(NULL)  # Return NULL if potential features could not be determined
    }
    
    # Select relevant columns and clean up
    selected_columns <- c(potential_features, "Activity", "Time", "ID")
    training_data_clean <- training_data[, ..selected_columns]
    training_data_clean <- na.omit(training_data_clean)  # Remove rows with NA values
    
    # Step 2: Check if there are multiple classes in the Activity column
    if (length(unique(training_data_clean$Activity)) <= 1) {
      message("Only one class, skipping Random Forest feature selection.")
      # Return all features if only one class exists
      return(training_data_clean)
    }
    
    # Step 3: Perform Random Forest feature selection if more than one class exists
    if (length(potential_features) <= number_features) {
      # If there are fewer potential features than the requested number, select all
      top_features <- c(potential_features, "Activity")
    } else {
      # Subsampling to reduce size of training data
      training_data_sampled <- training_data_clean %>%
        group_by(Activity, ID) %>%
        slice_sample(prop = 0.25) %>%
        ungroup()
      
      # Perform Random Forest feature selection
      feature_importance <- tryCatch({
        featureSelectionRF(
          data = training_data_sampled,
          n_trees = as.numeric(number_trees),
          number_features = as.numeric(number_features)
        )
      }, error = function(e) {
        message("Error in featureSelectionRF: ", e$message)
        return(NULL)
      })
      
      if (is.null(feature_importance)) {
        message("Feature selection failed, returning all features.")
        return(training_data_clean)
      }
      
      # Step 4: Select top features based on Random Forest results
      top_features <- tryCatch({
        feature_importance$Feature[1:number_features]
      }, error = function(e) {
        message("Error in selecting top features: ", e$message)
        return(NULL)
      })
      
      if (is.null(top_features)) {
        message("Top feature selection failed.")
        return(NULL)
      }
      
      top_features <- c(top_features, "Activity")  # Add Activity to the selected features
    }
    
    # Step 5: Select final features from the data
    selected_feature_data <- tryCatch({
      training_data_clean[, ..top_features]
    }, error = function(e) {
      message("Error in selecting final features: ", e$message)
      return(NULL)
    })
    
    return(selected_feature_data)
    
  }, error = function(e) {
    message("Error in featureSelection: ", e$message)
    return(NULL)
  })
}


# Random Forest Feature Selection with tryCatch ####
featureSelectionRF <- function(data, n_trees, number_features) {
  
  data <- data[complete.cases(data), ]
  target <- as.factor(data$Activity)
  numeric_features <- data[, .SD, .SDcols = setdiff(names(data), c("Time", "ID", "Activity"))]
  
  # Fit Random Forest model
  rf_model <- tryCatch({
    randomForest(
      x = numeric_features,
      y = target,
      ntree = n_trees,
      importance = TRUE
    )
  }, error = function(e) {
    message("Error in fitting Random Forest model: ", e$message)
    return(NULL)
  })
  
  # Extract and rank feature importance
  feature_importance <- tryCatch({
    importance(rf_model, type = 1) %>%
      as.data.frame() %>%
      rownames_to_column(var = "Feature") %>%
      rename(Importance = MeanDecreaseAccuracy) %>%
      arrange(desc(Importance))
  }, error = function(e) {
    message("Error in calculating feature importance: ", e$message)
    return(NULL)
  })
  
  if (is.null(feature_importance)) stop("Feature importance calculation failed.")
  
  # Step 6: Select top features
  top_features <- tryCatch({
    feature_importance[1:number_features, ]
  }, error = function(e) {
    message("Error in selecting top features: ", e$message)
    return(NULL)
  })
  
  if (is.null(top_features)) stop("Top feature selection failed.")
  
  # Return selected features
  return(top_features)
}


# Select Potential Features (Variance and Correlation Threshold) ####
removeBadFeatures <- function(feature_data, threshold) {
  
  # Step 1: Calculate variance for numeric columns
  numeric_columns <- feature_data[, .SD, .SDcols = !c("Activity", "Time", "ID")]
  variances <- numeric_columns[, lapply(.SD, var, na.rm = TRUE)]
  selected_columns <- names(variances)[!is.na(variances) & variances > threshold]
  
  # Step 2: Remove highly correlated features
  numeric_columns <- numeric_columns[, ..selected_columns]
  corr_matrix <- cor(numeric_columns, use = "pairwise.complete.obs")
  high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)
  remaining_features <- setdiff(names(numeric_columns), names(numeric_columns)[high_corr])
  
  return(remaining_features)
}