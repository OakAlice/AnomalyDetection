# Feature selection and dimensionality reduction

# Feature Selection using Random Forest ####
# Feature Selection using Random Forest with tryCatch ####
featureSelection <- function(training_data, number_trees, number_features) {
  tryCatch({
    # Step 1: Eliminate features with no variance and high correlation
    potential_features <- tryCatch({
      removeBadFeatures(training_data, threshold = 0.9)
    }, error = function(e) {
      message("Error in removeBadFeatures: ", e$message)
      return(NULL)
    })
    
    selected_columns <- c(potential_features, "Activity", "Time", "ID")
    
    # clean it up
    training_data <- training_data[, ..selected_columns]
    training_data <- training_data[complete.cases(training_data), ]
    
    # Check if potential features are less than or equal to number_features
    if (length(potential_features) <= number_features) {
      top_features <- c(potential_features, "Activity")
    } else {
      # Step 3: Perform feature selection using Random Forest
      feature_importance <- tryCatch({
        featureSelectionRF(
          data = training_data,
          n_trees = as.numeric(number_trees),
          number_features = as.numeric(number_features)
        )
      }, error = function(e) {
        stop("Error in featureSelectionRF: ", e$message)
      })
      
      # If feature importance is NULL, skip feature selection
      if (is.null(feature_importance)) stop("Feature selection failed.")
      
      # Step 4: Select top features based on Random Forest results
      top_features <- tryCatch({
        feature_importance$Feature[1:number_features]
      }, error = function(e) {
        message("Error in selecting top features: ", e$message)
        return(NULL)
      })
      
      top_features <- c(top_features, "Activity")
    }
    
    # Step 5: Select final features from the data
    selected_feature_data <- tryCatch({
      training_data[, ..top_features]
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


# UMAP REduction ####
UMAPReduction <- function(numeric_features, labels, minimum_distance, num_neighbours, shape_metric, spread) {
  
  #minimum_distance = 0.01
  #num_neighbours = 100
  #shape_metric = 'cosine'
  #spread = 3
  
  # Train UMAP model on the known data
  umap_model_2D <- umap::umap(numeric_features, 
                              n_neighbors = num_neighbours, 
                              min_dist = minimum_distance, 
                              metric = shape_metric,
                              spread = spread)
  
  umap_model_3D <- umap::umap(numeric_features, 
                              n_neighbors = num_neighbours, 
                              min_dist = minimum_distance, 
                              metric = shape_metric, 
                              spread = spread,
                              n_components = 3)
  
  # Apply the trained UMAP model on training data
  umap_result_2D <- umap_model_2D$layout
  umap_result_3D <- umap_model_3D$layout
  
  # Create dataframes for 2D and 3D embeddings, add labels back
  umap_df <- as.data.frame(umap_result_2D)
  colnames(umap_df) <- c("UMAP1", "UMAP2")
  umap_df$Activity <- labels[1:nrow(umap_df), ]
  
  umap_df_3 <- as.data.frame(umap_result_3D)
  colnames(umap_df_3) <- c("UMAP1", "UMAP2", "UMAP3")
  umap_df_3$Activity <- labels[1:nrow(umap_df_3), ]
  
  # Plot the clusters in 2D
  UMAP_2D_plot <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Activity)) +
    geom_point(alpha = 0.6) +
    theme_minimal() +
    labs(x = "Dimension 1", y = "Dimension 2", colour = "Activity") +
    theme(legend.position = "right") +
    annotate("text", x = Inf, y = -Inf, label = paste("n_neighbors:", num_neighbours, "\nmin_dist:", minimum_distance, "\nmetric:", shape_metric),
             hjust = 1.1, vjust = -0.5, size = 3, color = "black", fontface = "italic")+
    scale_color_discrete()
  
  # Plot in 3D
  UMAP_3D_plot <- plotly::plot_ly(umap_df_3, x = ~UMAP1, y = ~UMAP2, z = ~UMAP3, 
                                  color = ~Activity, colors = "Set1", 
                                  type = "scatter3d", mode = "markers",
                                  marker = list(size = 3, opacity = 0.5)) %>% 
    plotly::layout(scene = list(xaxis = list(title = "UMAP1"), yaxis = list(title = "UMAP2"), zaxis = list(title = "UMAP3")))
  
  return(list(
    UMAP_3D_plot = UMAP_3D_plot,
    UMAP_2D_plot = UMAP_2D_plot,
    UMAP_2D_model = umap_model_2D,
    UMAP_3D_model = umap_model_3D,
    UMAP_2D_embeddings = umap_df,
    UMAP_3D_embeddings = umap_df_3
  ))
}

