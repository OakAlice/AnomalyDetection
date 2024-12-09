# Feature selection and dimensionality reduction

featureSelection <- function(training_data, number_features, corr_threshold = 0.9) {
  tryCatch({
    # Step 1: Filter numeric features
    numeric_columns <- training_data %>%
      select(-Activity, -Time, -ID) %>%
      as.data.table()
    
    # Remove columns with >50% NA
    valid_cols <- names(numeric_columns)[colMeans(is.na(numeric_columns)) <= 0.5]
    numeric_columns <- numeric_columns[, ..valid_cols]
    
    # Remove zero-variance columns
    zero_variance <- numeric_columns[, sapply(.SD, function(col) sd(col, na.rm = TRUE) > 0)]
    zero_variance <- na.omit(zero_variance)
    numeric_columns <- numeric_columns[, names(zero_variance)[zero_variance], with = FALSE]
    
    # Remove highly correlated features
    if (ncol(numeric_columns) > 1) {
      corr_matrix <- cor(numeric_columns, use = "pairwise.complete.obs")
      high_corr <- findCorrelation(corr_matrix, cutoff = corr_threshold, names = TRUE)
      numeric_columns <- numeric_columns[, setdiff(names(numeric_columns), high_corr), with = FALSE]
    }
    
    # Check if any features remain
    if (ncol(numeric_columns) == 0) stop("No valid features remaining after preprocessing.")
    
    # Step 2: Clean training data
    training_data_clean <- cbind(numeric_columns, Activity = training_data$Activity) %>%
      na.omit() %>%
      as.data.table()
    
    # Check for multiple classes in Activity column
    if (length(unique(training_data_clean$Activity)) <= 1) {
      message("Only one class detected; skipping Random Forest feature selection.")
      flush.console()
      top_features <- names(numeric_columns)
    } else {
      # Random Forest feature selection
      if (ncol(numeric_columns) > number_features) {
        message("Starting Random Forest feature selection.")
        flush.console()
        
        # Sample 75% of data for feature selection
        sampled_data <- training_data_clean %>%
          group_by(Activity) %>%
          slice_sample(prop = 0.75) %>%
          as.data.frame()
        
        top_features <- featureSelectionRF(
          data = sampled_data,
          n_trees = 400,
          number_features = as.numeric(number_features)
        )
        
        if (is.null(top_features)) {
          message("Random Forest feature selection failed; returning all features.")
          flush.console()
          top_features <- names(numeric_columns)
        }
      } else {
        top_features <- names(numeric_columns)
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

featureSelectionRF <- function(data, n_trees, number_features) {
  tryCatch({
    data <- data %>%
      mutate(Activity = factor(Activity))

    # Fit Random Forest model with ranger
    rf_model <- ranger(
      dependent.variable.name = "Activity",
      data = data,
      num.trees = round(n_trees, 0),
      importance = "impurity",  # Equivalent to MeanDecreaseAccuracy
      classification = TRUE,
      probability = FALSE
    )
    
    # Extract and rank feature importance
    importance_df <- as.data.frame(rf_model$variable.importance) %>%
      tibble::rownames_to_column("Feature") %>%
      arrange(desc(rf_model$variable.importance)) %>%
      slice_head(n = min(number_features, nrow(.)))
    
    top_features <- importance_df$Feature
    
    return(top_features)
  }, error = function(e) {
    message("Error in Random Forest feature selection: ", e$message)
    return(NULL)
  })
}

