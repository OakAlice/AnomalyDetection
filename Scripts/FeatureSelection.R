# Feature selection and dimensionality reduction

# Feature Selection using Random Forest ####
featureSelection <- function(training_data, number_trees, number_features) {
  
  # Step 1: Eliminate features with no variance and high correlation
  potential_features <- removeBadFeatures(training_data, threshold = 0.9)
  selected_columns <- c(potential_features, "Activity", "Time", "ID")
  training_data <- training_data[, ..selected_columns]
  training_data <- training_data[complete.cases(training_data), ]
  
  # Step 2: Perform feature selection using Random Forest
  RF_features <- featureSelectionRF(
    data = training_data,
    target_column = "Activity",
    n_trees = as.numeric(number_trees),
    number_features = as.numeric(number_features)
  )
  
  # Step 3: Select top features and keep label columns
  top_features <- RF_features$Selected_Features$Feature[1:number_features]
  final_columns <- c(top_features, "Activity", "Time", "ID")
  selected_feature_data <- training_data[, ..final_columns]
  
  return(selected_feature_data)
}

# Random Forest Feature Selection ####
featureSelectionRF <- function(data, target_column, n_trees, number_features) {
  
  # Ensure complete cases and extract target
  data <- data[complete.cases(data), ]
  target <- as.factor(data[[target_column]])
  numeric_features <- data[, .SD, .SDcols = setdiff(names(data), c("Time", "ID", "Activity"))]
  
  # Fit Random Forest model
  rf_model <- randomForest(
    x = numeric_features,
    y = target,
    ntree = n_trees,
    importance = TRUE
  )
  
  # Extract and rank feature importance
  feature_importance <- importance(rf_model, type = 1) %>%
    as.data.frame() %>%
    rownames_to_column(var = "Feature") %>%
    rename(Importance = MeanDecreaseAccuracy) %>%
    arrange(desc(Importance))
  
  # Plot feature importance for the top features
  feature_importance_plot <- ggplot(feature_importance[1:number_features, ], 
                                    aes(x = reorder(Feature, -Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(
      title = "Top Feature Importance (Random Forest)",
      x = "Features",
      y = "Importance"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
  # Return selected features and the importance plot
  list(
    Selected_Features = feature_importance,
    Feature_Importance_Plot = feature_importance_plot
  )
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
