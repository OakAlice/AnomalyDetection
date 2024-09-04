# Feature selection and dimensionality reduction

# UMAP ####
UMAP_reduction <- function(feature_data, minimum_distance, num_neighbours, shape_metric){
  # select numeric columsn and remove all that have NA
  numeric_features <- feature_data %>%
    select_if(is.numeric) %>% # Select only numeric columns
    select(-Time, -ID, -X_nperiods, -X_seasonal_period, -Y_zero_proportion, -Y_nperiods, -Y_seasonal_period, -Z_nperiods, -Z_seasonal_period, -X_alpha, -X_beta, -X_gamma, -Y_alpha, -Y_beta, -Y_gamma, -Z_alpha, -Z_beta, -Z_gamma) %>% # these were all NA
    na.omit()
  
  umap_result_2D <- umap(numeric_features, n_neighbors = num_neighbours, min_dist = minimum_distance, metric = shape_metric)
  umap_result_3D <- umap(numeric_features, n_neighbors = num_neighbours, min_dist = minimum_distance, metric = shape_metric, n_components = 3)
  
  # dataframe, name, and add back identifying info
  umap_df <- as.data.frame(umap_result_2D$layout)
  colnames(umap_df) <- c("UMAP1", "UMAP2")
  umap_df$Activity <- feature_data$Activity[1:length(umap_df$UMAP1)]
  umap_df$ID <- feature_data$ID[1:length(umap_df$UMAP1)]  
  
  # plot the clusters in 2D
  ### Change to datashader rendering
  UMAP_2D_plot <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Activity)) +
    geom_point(alpha = 0.6) +
    theme_minimal() +
    labs(
      x = "Dimension 1",
      y = "Dimension 2",
      colour = "Activity"
    ) +
    theme(
      legend.position = "right"
    ) +
    annotate(
      "text",
      x = Inf, y = -Inf,  # Position the text in the lower right corner
      label = paste("n_neighbors:", num_neighbours, "\nmin_dist:", minimum_distance, "\nmetric:", shape_metric),
      hjust = 1.1, vjust = -0.5,  # Adjust text alignment
      size = 3,  # Adjust text size if necessary
      color = "black",  # Adjust text color if necessary
      fontface = "italic"
    )
  
  # plot in 3D
  UMAP_3D_plot <- plotly::plot_ly(umap_df_3, x = ~UMAP1, y = ~UMAP2, z = ~UMAP3, 
                                  color = ~Activity, colors = "Set1", 
                                  type = "scatter3d", mode = "markers",
                                  marker = list(size = 3, opacity = 0.5)) %>%  # Adjust size and opacity here
    plotly::layout(
      scene = list(
        xaxis = list(title = "UMAP1"),
        yaxis = list(title = "UMAP2"),
        zaxis = list(title = "UMAP3")
      )
    )
  
  return(list(UMAP_3D_plot = UMAP_3D_plot,
         UMAP_2D_plot = UMAP_2D_plot))
}  



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
RF_feature_selection <- function(feature_data, target_column, n_trees, number_features) {
  # Ensure the target column is a factor
  feature_data[[target_column]] <- as.factor(feature_data[[target_column]])
  
  # Extract numeric features and target variable
  numeric_features <- feature_data %>%
    select_if(is.numeric) %>% # Select only numeric columns
    select(-Time, -ID, -X_nperiods, -X_seasonal_period, -Y_zero_proportion, -Y_nperiods, -Y_seasonal_period, -Z_nperiods, -Z_seasonal_period, -X_alpha, -X_beta, -X_gamma, -Y_alpha, -Y_beta, -Y_gamma, -Z_alpha, -Z_beta, -Z_gamma) %>% # these were all NA
    na.omit()
  target <- feature_data[[target_column]]
  
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
      title = "Top 10 Feature Importance based on Random Forest",
      x = "Features",
      y = "Importance (Mean Decrease in Accuracy)"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
  
  # Plot 2: OOB Error Rate Plot
  oob_error_rate <- rf_model$err.rate[, 1]
  
  oob_error_plot <- ggplot(data.frame(Trees = 1:length(oob_error_rate), OOB_Error = oob_error_rate), aes(x = Trees, y = OOB_Error)) +
    geom_line(color = "blue", size = 1) +
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
