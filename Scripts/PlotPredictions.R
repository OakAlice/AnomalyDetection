
# Plot overlap ------------------------------------------------------------
library(patchwork)

# Binary ------------------------------------------------------------------

test_data_labels <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_multi_features.csv"))) %>%
  select(Activity, Time, ID) %>%
  filter(ID == ID[1])

files <- list.files(file.path(base_path, "Output", "Predictions"), "*?_binary_predictions.csv", full.names = TRUE)

data_files <- lapply(files, function(file) {
  df <- fread(file)
  df[, extracted_word := str_extract(basename(file), paste0("(?<=^", dataset_name, "_)[^_]+(?=_binary_predictions)"))]
  return(df)
})
data <- rbindlist(data_files)

data <- as.data.table(data) %>% group_by(ID) %>% arrange(Time) %>% mutate(rowcount = row_number())

# Define palette for Activities and predicted classes
full_activity_colors <- c(
  "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
  "#E07A5F", "#F2CC8F", "#81B29A", "#3D5A80", "#98C1D9", "#EE6C4D",
  "#6A0572", "#CC444B", "#80A1C1", "#B4A7D6"
)

specific_activity_colors <-  c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
class_colors <- c("-1" = "grey", "1" = "#FFCF56")

# Top graph: Ground truth activity tile plot
activity_plot <- ggplot(test_data_labels, aes(x = Time, y = 1, fill = Activity)) +
  geom_tile() +
  scale_fill_manual(values = full_activity_colors, name = "Activity") +
  labs(x = "Time", y = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank(),
        legend.position = "right")

# Bottom graph: Predictions by individual classification models
prediction_plot <- ggplot(data, aes(x = Time, y = extracted_word, fill = as.factor(predicted_classes))) +
  geom_tile() +
  scale_fill_manual(values = class_colors, name = "Predicted Class") +
  labs(x = "Time", y = "Predicted Model") +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(size = 10))

# Combine the plots: Top for ground truth, bottom faceted for predictions
combined_plot <- activity_plot / prediction_plot +
  plot_layout(heights = c(1, 3)) # Adjust height ratio for visibility

# Display the combined plot
combined_plot









# 1-class ------------------------------------------------------------------

i <- 5

test_data_labels <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_multi_features.csv")))
test_data_labels <- test_data_labels %>%
  select(Activity, Time, ID) %>%
  filter(ID == unique(test_data_labels$ID)[i])

selected_ID <- test_data_labels$ID[1]

files <- list.files(file.path(base_path, "Output", "Predictions"), "*?_OCC_predictions.csv", full.names = TRUE)

data_files <- lapply(files, function(file) {
  df <- fread(file)
  df[, extracted_word := str_extract(basename(file), paste0("(?<=^", dataset_name, "_)[^_]+(?=_OCC_predictions)"))]
  df <- df[, -3, with = FALSE]  # Duplicate activity
  return(df)
})
data <- rbindlist(data_files)

data <- as.data.frame(data) %>%
  filter(ID == selected_ID) %>%
  arrange(Time) %>%
  mutate(condition = case_when(
    Activity == 1 & predicted_classes == 1 ~ "True Positive",
    Activity == -1 & predicted_classes == 1 ~ "False Positive",
    Activity == 1 & predicted_classes == -1 ~ "False Negative",
    Activity == -1 & predicted_classes == -1 ~ "True Negative",
    TRUE ~ "Not predicted"
  ))

# Define palette for Activities and predicted classes
full_activity_colors <- c(
  "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
  "#E07A5F", "#F2CC8F", "#81B29A", "#3D5A80", "#98C1D9", "#EE6C4D",
  "#6A0572", "#CC444B", "#80A1C1", "#B4A7D6"
)

specific_activity_colors <-  c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
condition_colours <- c("True Positive" = "darkgreen", "True Negative" = "lightgreen", "False Positive" = "orange", "False Negative" = "red", "grey")

# Top graph: Ground truth activity tile plot
activity_plot <- ggplot(test_data_labels, aes(x = Time, y = 1, fill = Activity)) +
  geom_tile() +
  scale_fill_manual(values = full_activity_colors, name = "Activity") +
  labs(x = "Time", y = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        panel.grid = element_blank(),
        legend.position = "right")

# Bottom graph: Predictions by individual classification models
prediction_plot <- ggplot(data, aes(x = Time, y = extracted_word, fill = as.factor(condition))) +
  geom_tile() +
  scale_fill_manual(values = condition_colours, name = "Predicted Class") +
  labs(x = "Time", y = "Predicted Model") +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(size = 10))

# Combine the plots: Top for ground truth, bottom faceted for predictions
combined_plot <- activity_plot / prediction_plot
combined_plot

