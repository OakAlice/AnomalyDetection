# UMAP visualisation


# UMAP ####
UMAP_reduction <- function(numeric_features, labels, minimum_distance, num_neighbours, shape_metric, save_model_path = NULL) {
  # Train UMAP model on the known data
  umap_model_2D <- umap::umap(numeric_features, n_neighbors = num_neighbours, min_dist = minimum_distance, metric = shape_metric)
  umap_model_3D <- umap::umap(numeric_features, n_neighbors = num_neighbours, min_dist = minimum_distance, metric = shape_metric, n_components = 3)
  
  # Save the trained UMAP models for future use (for transforming new data)
  if (!is.null(save_model_path)) {
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
  umap_df$Activity <- as.factor(umap_df$Activity$Activity)
  
  umap_df_3 <- as.data.frame(umap_result_3D)
  colnames(umap_df_3) <- c("UMAP1", "UMAP2", "UMAP3")
  umap_df_3$Activity <- labels[1:nrow(umap_df_3), ]
  umap_df_3$Activity <- as.factor(umap_df_3$Activity$Activity)
  
  # Plot the clusters in 2D
  UMAP_2D_plot <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, colour = Activity)) +
    geom_point(alpha = 0.6) +
    theme_minimal() +
    labs(x = "Dimension 1", y = "Dimension 2", colour = "Activity") +
    theme(legend.position = "right") +
    annotate("text", x = Inf, y = -Inf, label = paste("n_neighbors:", num_neighbours, "\nmin_dist:", minimum_distance, "\nmetric:", shape_metric),
             hjust = 1.1, vjust = -0.5, size = 3, color = "black", fontface = "italic") +
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

#UMAP_transform_new_data <- function(new_data, umap_model_path) {
#  umap_model <- readRDS(umap_model_path)
#  transformed_data <- predict(umap_model, new_data)
#  return(transformed_data)
#}
