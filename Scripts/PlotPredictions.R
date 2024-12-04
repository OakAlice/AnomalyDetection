
# Plot overlap ------------------------------------------------------------

# read in the data

# Binary ------------------------------------------------------------------

files <- list.files(file.path(base_path, "Output", "Predictions"), "*?_binary_predictions.csv", full.names = TRUE)

data_files <- lapply(files, function(file) {
  df <- fread(file)
  df[, extracted_word := str_extract(basename(file), paste0("(?<=^", dataset_name, "_)[^_]+(?=_binary_predictions)"))]
  return(df)
})
data <- rbindlist(data_files)

data <- as.data.table(data) %>% group_by(ID) %>% arrange(Time) %>% mutate(rowcount = row_number())

# Define palette for Activities and predicted classes
activity_colors <-  c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
class_colors <- c("-1" = "grey", "1" = "#FFCF56")

# Top graph: Ground truth activity tile plot
activity_plot <- ggplot(data, aes(x = rowcount, y = 1, fill = Activity)) +
  geom_tile() +
  scale_fill_manual(values = activity_colors, name = "Activity") +
  labs(title = "Ground Truth Behaviors Over Time", x = "Time", y = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank(),
        legend.position = "bottom")

# Separate predictions by extracted_word for independent plots
prediction_plots <- data %>%
  split(.$extracted_word) %>%
  lapply(function(df) {
    ggplot(df, aes(x = rowcount, y = 1, fill = as.factor(predicted_classes))) +
      geom_tile() +
      scale_fill_manual(values = class_colors, name = "Predicted Class") +
      labs(title = paste("Predictions for", unique(df$extracted_word)),
           x = "Time",
           y = NULL) +
      theme_minimal() +
      theme(axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            panel.grid = element_blank(),
            legend.position = "bottom")
  })

# Combine the activity plot with all prediction plots
combined_plot <- activity_plot / wrap_plots(prediction_plots, ncol = 1)

# Display the combined plot
print(combined_plot)

