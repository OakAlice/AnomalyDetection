#---------------------------------------------------------------------------
# Explore data to determine target behaviours and window length         ####
#---------------------------------------------------------------------------


#---------------------------------------------------------------------------
# Find target behaviours                                                ####
#---------------------------------------------------------------------------
# set colours
generate_random_colors <- function(n) {
  colors <- rgb(runif(n), runif(n), runif(n))
  return(colors)
}

## total volume by Activity and ID ####
plotActivityByID <- function(data, frequency, colours) {
  my_colours <- generate_random_colors(colours)
  # summarise into a table
  labelledDataSummary <- data %>%
    #filter(!Activity %in% ignore_behaviours) %>%
    count(ID, Activity)
  
  # account for the HZ, convert to minutes
  labelledDataSummaryplot <- labelledDataSummary %>%
    mutate(minutes = (n/frequency)/60)
  
  # Plot the stacked bar graph
  plot_activity_by_ID <- ggplot(labelledDataSummaryplot, aes(x = Activity, y = minutes, fill = as.factor(ID))) +
    geom_bar(stat = "identity") +
    labs(x = "Activity",
         y = "minutes") +
    theme_minimal() +
    scale_fill_manual(values = my_colours) +
    theme(axis.line = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.border = element_rect(color = "black", fill = NA),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  return(plot_activity_by_ID)
}

## examples of each behaviour trace ####
plotTraceExamples <- function(behaviours, data, individuals, n_samples, n_col) {
  
  data <- data %>% filter(ID %in% sample(unique(data$ID), individuals))
  
  # Create plots for each behavior (with error catching)
  plots <- purrr::map(behaviours, function(behaviour) {
    tryCatch(
      {
        plot_behaviour(behaviour, n_samples, data)
      },
      error = function(e) {
        message("Skipping plot for ", behaviour, ": ", e$message)
        NULL  # Return NULL to indicate skipping
      }
    )
  })
  
  # Remove NULL plots (for behaviors with no data)
  plots <- purrr::compact(plots)
  
  # Combine plots into a single grid
  grid_plot <- cowplot::plot_grid(plotlist = plots, ncol = n_col)
  
  return(grid_plot)
}

# Function to create the plot for each behavior
plot_behaviour <- function(behaviour, n_samples, data) {
  df <- data %>%
    filter(Activity == behaviour) %>%
    group_by(ID, Activity) %>%
    slice(1:n_samples) %>%
    mutate(relative_time = row_number())
  
  # Check if the filtered dataframe is empty
  if (nrow(df) == 0) {
    stop("No data available for behaviour: ", behaviour)
  }
  
  ggplot(df, aes(x = relative_time)) +
    geom_line(aes(y = Accelerometer.X, color = "X"), show.legend = FALSE) +
    geom_line(aes(y = Accelerometer.Y, color = "Y"), show.legend = FALSE) +
    geom_line(aes(y = Accelerometer.Z, color = "Z"), show.legend = FALSE) +
    labs(title = paste(behaviour),
         x = NULL, y = NULL) +
    scale_color_manual(values = c(X = "salmon", Y = "turquoise", Z = "darkblue"), guide = "none") +
    facet_wrap(~ ID, nrow = 1, scales = "free_x") +
    theme_minimal() +
    theme(panel.grid = element_blank(),
          axis.text.x = element_blank(),
          axis.text.y = element_blank())
}

#---------------------------------------------------------------------------
# Find duration of target behaviours                                    ####
#---------------------------------------------------------------------------
BehaviourDuration <- function(data, sample_rate, target_activities){
  
  summary <- data %>%
    mutate(Activity = ifelse(Activity %in% target_activities, Activity, "Other")) %>%
    arrange(ID) %>%                     # Sort by ID (not time because multiple trials in dog data)
    group_by(ID) %>%                    # Group by ID
    mutate(
      behavior_change = lag(Activity) != Activity,  # Detect changes in Activity
      behavior_change = ifelse(is.na(behavior_change), TRUE, behavior_change)  # Handle the first row
    ) %>%
    mutate(
      behavior_id = cumsum(behavior_change)  # Create an identifier for each continuous behavior segment
    ) %>%
    group_by(ID, behavior_id) %>%            # Group by ID and behavior_id
    mutate(
      row_count = row_number()                # Count rows within each behavior segment
    ) %>%
    ungroup() %>%
    select(ID, Time, Activity, row_count, behavior_id) %>%   # Select relevant columns
    group_by(ID, Activity, behavior_id) %>%
    summarise(duration_sec = max(row_count)/100) %>%
    filter(Activity %in% target_activities)
  
  duration_stats <- summary %>%
    group_by(Activity) %>%
    summarise(
      average_behaviour_duration = mean(duration_sec, na.rm = TRUE),
      median_behaviour_duration = median(duration_sec, na.rm = TRUE),
      variation_duration = sd(duration_sec, na.rm = TRUE),
      maximum_duration = max(duration_sec, na.rm = TRUE),
      minimum_duration = min(duration_sec, na.rm = TRUE),
      third_quartile_duration = quantile(duration_sec, 0.25, na.rm = TRUE),  # Add third quartile
      count = n()
    )
  
  # plot that
  duration_plot <- ggplot(summary, aes(x = Activity, y = as.numeric(duration_sec))) +
    geom_boxplot(aes(color = Activity)) +  # Use color to distinguish activities
    theme_minimal() +
    theme(
      legend.position = "none",             # Remove legend
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),  # Rotate x-axis labels 90 degrees
      panel.grid = element_blank(),         # Remove grid lines
      panel.border = element_rect(color = "black", fill = NA)  # Add black border around the plot
    ) +
    labs(
      x = "Activity",
      y = "Duration (seconds)"
    ) +
    scale_y_continuous(
      limits = c(min(summary$duration_sec, na.rm = TRUE), max(summary$duration_sec, na.rm = TRUE)),  # Set y-axis limits
      breaks = seq(0, max(summary$duration_sec, na.rm = TRUE), by = 160)  # Adjust the step size as needed
    )
  
  return(list(duration_plot = duration_plot,
              duration_stats = duration_stats))
}

# how behaviours change over time ####

# plot how the behaviours change over time
data2 <- data_other %>% arrange(ID, Timestamp) %>% slice_head(n = 940000) %>% ungroup() %>%
  mutate(numeric_activity = as.numeric(factor(Activity)), 
         Activity = factor(Activity),
         relative_seconds = row_number())

ggplot(data2, aes(x = (relative_seconds/20), y = as.numeric(numeric_activity))) +
  geom_line() +
  theme_minimal() +
  labs(
    x = "Time (seconds)",
    y = "Activity",
    color = "Activity"
  ) +
  scale_y_continuous(
    breaks = unique(data2$numeric_activity),
    labels = levels(data2$Activity)
  )









#---------------------------------------------------------------------------
# Plotting features                                                     ####
#---------------------------------------------------------------------------

# plot the training_data features ####
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

plotFeatureDistribution <- function(data_processed) {
  # Reshape data from wide to long format
  data_long <- data_processed %>%
    pivot_longer(
      cols = starts_with("X_") | starts_with("Y_") | starts_with("Z_"),
      names_to = "Feature",
      values_to = "Value"
    )
  
  # Create the plot
  ggplot(data_long, aes(x = Value, fill = Activity)) +
    geom_histogram(position = "identity", alpha = 0.6, binwidth = 0.1) +
    facet_wrap(~ Feature, scales = "free_x", ncol = 4) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),  # Remove x-axis text
      axis.text.y = element_blank(),  # Remove y-axis text
      axis.title.x = element_blank(), # Remove x-axis title
      axis.title.y = element_blank(), # Remove y-axis title
      axis.ticks = element_blank()    # Remove axis ticks
    ) +
    labs(
      fill = "Activity"
    )
}


# same idea as above but different view ####
data_long <- data_processed %>%
  pivot_longer(cols = -c(Activity, Timestamp, ID), names_to = "feature", values_to = "value")


data_long_ind <- data_long %>% filter(ID %in% 22)

# Plot using ggplot
ggplot(data_long_ind, aes(x = Activity, y = value, colour = Activity)) +
  geom_jitter() +
  facet_wrap(~ feature, scales = "free_y") +
  theme(axis.text.x = element_blank()) +
  theme_minimal()+
  labs(title = "Jitter Plot of Features by Activity",
       x = "Activity",
       y = "Value")


# interactions between the features ####
library(GGally)

# Select the numeric columns for pairwise plotting
numeric_data <- data_processed %>%
  select(-c(Activity, Timestamp, ID
            )) %>%
  filter_all(all_vars(is.finite(.))) %>%
  na.omit()

# Pairwise scatter plots using ggpairs
numeric_columns <- colnames(numeric_data)

# Create pairwise scatter plots
for (i in 1:(length(numeric_columns) - 1)) {
  for (j in (i + 1):length(numeric_columns)) {
    p <- ggplot(data_processed, aes_string(x = numeric_columns[i], y = numeric_columns[j], colour = "Activity")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(title = paste("Scatter Plot of", numeric_columns[i], "vs", numeric_columns[j]),
           x = numeric_columns[i],
           y = numeric_columns[j])
    
    print(p)
  }
}



## Here ####
library(patchwork)
data_long <- data_processed %>%
  select(-c(Timestamp, ID)) %>%
  gather(key = "feature", value = "value", -Activity)

features <- unique(data_long$feature)

# Initialize an empty plot object
plot_list <- list()

# Iterate through each pair of features
for(i in 1:(length(features)-1)) {
  for(j in (i+1):length(features)) {
    plot <- ggplot(data_processed, aes_string(x = features[i], y = features[j], colour = "Activity")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(x = features[i], y = features[j]) +
      theme(legend.position = "none")
    
    plot_list[[paste(features[i], features[j], sep = "_vs_")]] <- plot
  }
}

# Combine all plots into a grid using patchwork
combined_plot <- wrap_plots(plot_list, ncol = 3) +
  plot_layout(guides = "collect") +
  plot_annotation(title = "Pairwise Scatter Plots of Features by Activity")

# Display the plot
print(combined_plot)



# ---------------------------------------------------------------------------
# UMAP ####
# ---------------------------------------------------------------------------


# UMAP visualisation
# UMAP ####
UMAPReduction <- function(numeric_features, labels, minimum_distance, num_neighbours, shape_metric, save_model_path = NULL) {
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
