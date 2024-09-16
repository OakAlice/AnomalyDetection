# Feature selection and dimensionality reduction

# UMAP ####
UMAP_reduction <- function(numeric_features, labels, minimum_distance, num_neighbours, shape_metric, save_model_path) {
  
  # minimum_distance = 0.1
  # num_neighbours = 40
  # metric = "euclidean"
  
  # Train UMAP model on the known data
  umap_model_2D <- umap::umap(numeric_features, n_neighbors = num_neighbours, min_dist = minimum_distance, metric = shape_metric)
  umap_model_3D <- umap::umap(numeric_features, n_neighbors = num_neighbours, min_dist = minimum_distance, metric = shape_metric, n_components = 3)
  
  # Save the trained UMAP models for future use (for transforming new data)
  if (!is.null(save_model_path)) {
    ensure.dir(save_model_path)
    saveRDS(umap_model_2D, file = file.path(save_model_path, "umap_2D_model.rds"))
    saveRDS(umap_model_3D, file = file.path(save_model_path, "umap_3D_model.rds"))
  }
  
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
  UMAP_2D_plot <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Activity$Activity)) +
    geom_point(alpha = 0.6) +
    theme_minimal() +
    labs(x = "Dimension 1", y = "Dimension 2", colour = "Activity") +
    theme(legend.position = "right") +
    annotate("text", x = Inf, y = -Inf, label = paste("n_neighbors:", num_neighbours, "\nmin_dist:", minimum_distance, "\nmetric:", shape_metric),
             hjust = 1.1, vjust = -0.5, size = 3, color = "black", fontface = "italic")+
    scale_color_discrete()
  
  # Plot in 3D
  UMAP_3D_plot <- plotly::plot_ly(umap_df_3, x = ~UMAP1, y = ~UMAP2, z = ~UMAP3, 
                                  color = ~Activity$Activity, colors = "Set1", 
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

#UMAP_transform_new_data <- function(new_data, umap_model_path) {
#  umap_model <- readRDS(umap_model_path)
#  transformed_data <- predict(umap_model, new_data)
#  return(transformed_data)
#}

# PCA ####

PCA_reduction <- function(feature_data) {
  # Ensure that the data contains only numeric columns
  numeric_features <- feature_data %>%
    select_if(is.numeric) %>% # Select only numeric columns
    select(-Time, -ID, -X_nperiods, -X_seasonal_period, -Y_zero_proportion, -Y_nperiods, -Y_seasonal_period, -Z_nperiods, -Z_seasonal_period, -X_alpha, -X_beta, -X_gamma, -Y_alpha, -Y_beta, -Y_gamma, -Z_alpha, -Z_beta, -Z_gamma) %>% # these were all NA
    na.omit()  # Remove any rows with NA values
  
  # Perform PCA
  pca_result <- prcomp(numeric_features, center = TRUE, scale. = FALSE)
  
  # Create a dataframe for the PCA results
  pca_df <- as.data.frame(pca_result$x)
  
  # Add back identifying information
  pca_df$Activity <- feature_data$Activity[1:nrow(pca_df)]
  pca_df$ID <- feature_data$ID[1:nrow(pca_df)]
  
  # Return the PCA result and the transformed data
  list(pca_result = pca_result, 
       pca_df = pca_df)
}

# Scree Plot
scree_plot <- function(pca_result) {
  # Variance explained by each principal component
  explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)
  
  # Create a dataframe for the scree plot
  scree_df <- data.frame(
    Principal_Component = paste0("PC", 1:length(explained_variance)),
    Variance_Explained = explained_variance
  )
  
  # Select the top 10 most explanatory components
  scree_df_top10 <- scree_df %>%
    arrange(desc(Variance_Explained)) %>%
    head(10) %>%
    arrange(Variance_Explained) # Re-order for better visualization
  
  # Create a column chart
  ggplot(scree_df_top10, aes(x = reorder(Principal_Component, Variance_Explained), y = Variance_Explained)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    labs(
      title = "Top 10 Principal Components by Variance Explained",
      x = "Principal Component",
      y = "Proportion of Variance Explained"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) # Rotate x-axis labels
}

scatter_plot_pca <- function(pca_df) {
  ggplot(pca_df, aes(x = PC1, y = PC2, color = Activity)) +
    geom_point(alpha = 0.6) +
    labs(
      title = "PCA Scatter Plot (PC1 vs PC2)",
      x = "Principal Component 1",
      y = "Principal Component 2",
      color = "Activity"
    ) +
    theme_minimal()
}



# LDA ####
LDA_feature_selection <- function(feature_data, target_column, number_features) {
  # Ensure target column is a factor
  feature_data[[target_column]] <- as.factor(feature_data[[target_column]])
  
  # Extract numeric features and target variable
  numeric_features <- feature_data %>%
    select_if(is.numeric) %>% # Select only numeric columns
    select(-Time, -ID, -X_nperiods, -X_seasonal_period, -Y_zero_proportion, -Y_nperiods, -Y_seasonal_period, -Z_nperiods, -Z_seasonal_period, -X_alpha, -X_beta, -X_gamma, -Y_alpha, -Y_beta, -Y_gamma, -Z_alpha, -Z_beta, -Z_gamma) %>% # these were all NA
    na.omit()
  target <- feature_data[[target_column]]
  
  # Perform LDA
  lda_model <- MASS::lda(numeric_features, grouping = target)
  
  # Extract the linear discriminants
  lda_coefficients <- as.data.frame(lda_model$scaling)
  
  # Sort features by their absolute coefficients for the first linear discriminant
  selected_features <- lda_coefficients %>%
    rownames_to_column(var = "Feature") %>%
    mutate(Importance = abs(LD1)) %>%
    arrange(desc(Importance)) %>%
    select(Feature, Importance)
  
  # Plot 1: Feature Importance Bar Plot
  feature_importance_plot <- ggplot(selected_features[1:number_features, ], aes(x = reorder(Feature, -Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(
      title = "Feature Importance based on LDA",
      x = "Features",
      y = "Importance"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
  
  # Predict using LDA to get LD values for the data
  lda_values <- predict(lda_model, numeric_features)$x
  
  # Combine LDA values with the target column
  lda_df <- as.data.frame(lda_values) %>%
    mutate(Activity = feature_data[[target_column]])
  
  # Return the selected features and plots
  list(
    Selected_Features = selected_features,
    Feature_Importance_Plot = feature_importance_plot
  )
}

# Random Forest ####
RF_feature_selection <- function(data, target_column, n_trees, number_features) {
  
  # Extract numeric features and target variable
  data <- data[complete.cases(data), ]
  target <- as.factor(data[[target_column]])
  numeric_features <- data[, .SD, .SDcols = setdiff(names(data), c("Time", "ID", "Activity"))]
  
  # Fit Random Forest model
  rf_model <- randomForest(x = numeric_features, y = target, ntree = n_trees, importance = TRUE)
  
  # Extract feature importance
  feature_importance <- as.data.frame(importance(rf_model, type = 1)) %>%
    rownames_to_column(var = "Feature") %>%
    rename(Importance = MeanDecreaseAccuracy) %>%
    arrange(desc(Importance))
  
  # Plot 1: Feature Importance Bar Plot
  feature_importance_plot <- ggplot(feature_importance[1:number_features, ], aes(x = reorder(Feature, -Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "forestgreen") +
    coord_flip() +
    labs(
      title = "Top Feature Importance based on Random Forest",
      x = "Features",
      y = "Importance (Mean Decrease in Accuracy)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
  # Plot 2: OOB Error Rate Plot
  oob_error_rate <- rf_model$err.rate[, 1]
  
  oob_error_plot <- ggplot(data.frame(Trees = 1:length(oob_error_rate), OOB_Error = oob_error_rate), aes(x = Trees, y = OOB_Error)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(
      title = "Out-of-Bag (OOB) Error Rate vs. Number of Trees",
      x = "Number of Trees",
      y = "OOB Error Rate"
    ) +
    theme_minimal()
  
  # Return the selected features and plots
  list(
    Selected_Features = feature_importance,
    Feature_Importance_Plot = feature_importance_plot,
    OOB_Error_Plot = oob_error_plot
  )
}


  # select potential features 
select_potential_features <- function(feature_data, threshold) {
  
  # Select only the features (numeric columns)
  numeric_columns <- subset_data[, .SD, .SDcols = !c("Activity", "Time", "ID")]
  
  # Remove features with little variance (using a threshold)
  variances <- numeric_columns[, lapply(.SD, var, na.rm = TRUE)]  # Calculate variances
  selected_columns <- names(variances)[!is.na(variances) & variances > threshold]
  numeric_columns <- numeric_columns[, ..selected_columns]
  
  # Remove features highly correlated to others
  corr_matrix <- cor(numeric_columns, use = "pairwise.complete.obs")
  
  # Find correlated features
  high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)  # Identify highly correlated features (cutoff at 0.9)
  remaining_features <- setdiff(names(numeric_columns), names(numeric_columns)[high_corr])
  numeric_columns <- numeric_columns[, ..remaining_features]
  
  # Return the names of potential features
  potential_features <- names(numeric_columns)
  
  return(potential_features)
}
