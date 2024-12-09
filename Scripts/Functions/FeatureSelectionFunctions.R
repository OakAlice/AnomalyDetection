# Feature selection and dimensionality reduction

featureSelection <- function(training_data, number_trees, number_features) {
  tryCatch({
    
    # Step 1: Filter numeric features
    numeric_columns <- training_data[, .SD, .SDcols = !c("Activity", "Time", "ID")] %>% as.data.table()
    
    # Remove columns with >50% NA
    na_threshold <- 0.5
    na_fraction <- colMeans(is.na(numeric_columns))
    valid_cols <- names(na_fraction[na_fraction <= na_threshold])
    
    if (length(valid_cols) == 0) {
      stop("All numeric columns exceed the NA threshold.")
    }
    selected_numeric_columns <- numeric_columns[, ..valid_cols]
    
    # Remove zero-variance columns
    zero_variance <- sapply(selected_numeric_columns, function(col) sd(col, na.rm = TRUE) == 0)
    valid_cols <- names(zero_variance[!zero_variance])
    if (length(valid_cols) == 0) {
      stop("All numeric columns have zero variance.")
    }
    numeric_columns <- numeric_columns[, ..valid_cols]
    
    # Remove highly correlated features
    if (ncol(numeric_columns) > 1) {
      corr_matrix <- cor(numeric_columns, use = "pairwise.complete.obs")
      high_corr <- findCorrelation(corr_matrix, cutoff = 0.9, names = TRUE)
      remaining_features <- setdiff(names(numeric_columns), high_corr)
    } else {
      remaining_features <- names(numeric_columns)
    }
    
    if (length(remaining_features) == 0) {
      stop("No features remain after removing highly correlated columns.")
    }
    
    # Step 2: Clean training data
    training_data_clean <- training_data[, .SD, .SDcols = c(remaining_features, "Activity")]
    training_data_clean <- na.omit(training_data_clean)
    
    # message("features cleaned")
    
    # Step 3: Check for multiple classes in Activity column
    if (length(unique(training_data_clean$Activity)) <= 1) {
      message("Only one class detected; skipping Random Forest feature selection.")
      top_features <- remaining_features
    } else {
      # Step 4: Random Forest feature selection
      if (length(remaining_features) > number_features) {
        message("Starting Random Forest feature selection.")
        sampled_data <- training_data_clean %>%
          group_by(Activity) %>%
          slice_sample(prop = 0.25) %>%
          ungroup() %>%
          as.data.frame()
        
        feature_importance <- tryCatch({
          featureSelectionRF(
            data = sampled_data,
            n_trees = as.numeric(number_trees),
            number_features = as.numeric(number_features)
          )
        }, error = function(e) {
          message("Error in featureSelectionRF: ", e$message)
          return(NULL)
        })
        
        if (!is.null(feature_importance)) {
          top_features <- feature_importance$Feature[1:number_features]
        } else {
          message("Random Forest feature selection failed; returning all features.")
        }
      }
    }
    
    # Include "Activity" in the final feature set
    top_features <- c(top_features, "Activity")
    
    # Return the final dataset with selected features
    return(training_data_clean[, ..top_features])
    
  }, error = function(e) {
    message("Error in featureSelection: ", e$message)
    return(NULL)
  })
}

# Random Forest Feature Selection with tryCatch ####
featureSelectionRF <- function(data, n_trees, number_features) {
  
  data <- data[complete.cases(data), ]
  target <- as.factor(data$Activity)
  numeric_features <- data[, !(names(data) %in% "Activity")]
  
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
