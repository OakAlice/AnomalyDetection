library(patchwork)
library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(MLmetrics)

# Load and process OCC predictions --------------------------------------------
occ_files <- list.files(file.path(base_path, "Output", "Testing", "Predictions"), 
                        pattern = paste0(dataset_name, ".*_OCC_.*\\.csv$"), 
                        full.names = TRUE)

occ_data <- lapply(occ_files, function(file) {
  df <- fread(file)
  df[, model_activity := str_extract(basename(file), "(?<=OCC_)[^_]+(?=_predictions)")]
  return(df)
}) %>% rbindlist()

# Load and process Binary predictions ----------------------------------------
binary_files <- list.files(file.path(base_path, "Output", "Testing", "Predictions"), 
                          pattern = paste0(dataset_name, ".*_Binary_.*\\.csv$"), 
                          full.names = TRUE)

binary_data <- lapply(binary_files, function(file) {
  df <- fread(file)
  df[, model_activity := str_extract(basename(file), "(?<=Binary_)[^_]+(?=_predictions)")]
  return(df)
}) %>% rbindlist()

# Load and process Multi predictions -----------------------------------------
multi_files <- list.files(file.path(base_path, "Output", "Testing", "Predictions"), 
                         pattern = paste0(dataset_name, "_(Activity|OtherActivity|GeneralisedActivity)_predictions\\.csv$"), 
                         full.names = TRUE)

multi_data <- lapply(multi_files, function(file) {
  df <- fread(file)
  df[, model_type := str_extract(basename(file), "(Activity|OtherActivity|GeneralisedActivity)")]
  return(df)
}) %>% rbindlist()

# Define color palettes ----------------------------------------------------
occ_colors <- c("TRUE" = "#FFCF56", "FALSE" = "grey")
binary_colors <- c("1" = "#FFCF56", "-1" = "grey")
activity_colors <- c(
  "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
  "#E07A5F", "#F2CC8F", "#81B29A", "#3D5A80", "#98C1D9", "#EE6C4D",
  "#6A0572", "#CC444B", "#80A1C1", "#B4A7D6", "goldenrod"
)

# Create plots ------------------------------------------------------------

# OCC Plot
occ_plot <- ggplot(occ_data, aes(x = Time, y = model_activity, fill = Predictions)) +
  geom_tile() +
  scale_fill_manual(values = occ_colors, name = "Prediction") +
  labs(title = "OCC Model Predictions", x = "Time", y = "Target Activity") +
  theme_minimal() +
  theme(panel.grid = element_blank())

# Binary Plot
binary_plot <- ggplot(binary_data, aes(x = Time, y = model_activity, fill = as.factor(Predictions))) +
  geom_tile() +
  scale_fill_manual(values = binary_colors, name = "Prediction") +
  labs(title = "Binary Model Predictions", x = "Time", y = "Target Activity") +
  theme_minimal() +
  theme(panel.grid = element_blank())

# Multi Plots (one for each type)
multi_plots <- lapply(unique(multi_data$model_type), function(type) {
  data_subset <- multi_data[model_type == type]
  ggplot(data_subset, aes(x = Time, y = 1, fill = Predictions)) +
    geom_tile() +
    scale_fill_manual(values = activity_colors, name = "Activity") +
    labs(title = paste("Multi Model -", type), x = "Time", y = NULL) +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank()
    )
})

# Combine all plots # occ_plot / binary_plot /
combined_plot <- ( multi_plots[[1]] / multi_plots[[2]] / multi_plots[[3]]) +
  plot_layout(heights = c(2, 2, 1, 1, 1)) +
  plot_annotation(
    title = paste("Predictions for", dataset_name),
    theme = theme(plot.title = element_text(hjust = 0.5))
  )

# Display the combined plot
combined_plot
