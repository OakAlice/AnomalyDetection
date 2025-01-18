# Plotting the performance comparing dichotomous with multiclass ----------

# training_set <- "all"

# Read in -----------------------------------------------------------------
combined_results <- fread(file = file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", ML_method, "_complete.csv"))) %>% as.data.frame()
combined_results <- combined_results %>% mutate(Activity = ifelse(Activity == "Weightedmacroaverage", "Macroaverage", Activity)) %>%
  mutate(Dataset = as.factor(Dataset),
         training_set = as.factor(training_set),
         Model = as.factor(Model),
         Activity = as.factor(Activity))


# Select the variables I will use ------------------------------------------
remainder_activities <- setdiff(unique(combined_results$Activity), c(str_to_title(target_activities), "Macroaverage", "Other"))
activity_levels <- if(dataset_name == "Vehkaoja_Dog") {
  c("Macroaverage", "Walking", "Eating", "Lying Chest", "Shaking", "Other")
} else if(dataset_name == "Ladds_Seal") {
  c("Macroaverage", "Swimming", "Chewing", "Still", "Facerub", "Other",
    remainder_activities) 
} else {
  stop("Unknown dataset name")
}

selected_results <- combined_results %>%
  select(Dataset, training_set, Model, Activity, 
         Prevalence, FalsePositives, 
         Rand_adj_F1_Score_equal, Rand_adj_Precision_equal, Rand_adj_Recall_equal, Rand_adj_Accuracy_equal
         ) %>%
  rename(Adj_F1 = Rand_adj_F1_Score_equal,
         Adj_Precision = Rand_adj_Precision_equal,
         Adj_Recall = Rand_adj_Recall_equal,
         Adj_Accuracy = Rand_adj_Accuracy_equal
         ) %>%
  filter(Model %in% c("OCC_Ensemble", "Binary_Ensemble", "Activity", "OtherActivity")) %>%
  mutate(Model = case_when(
    # Model == "OCC" ~ "1-class",
    # Model == "Binary" ~ "Binary", 
    Model == "OCC_Ensemble" ~ "1-class ensemble",
    Model == "Binary_Ensemble" ~ "Binary ensemble",
    Model == "Activity" ~ "Full multi-class",
    Model == "OtherActivity" ~ "Other multi-class",
    # Model == "GeneralisedActivity" ~ "Generalised multi-class",
    TRUE ~ Model
  )) %>%
  mutate(
    Activity = factor(Activity, levels = activity_levels),
    Model = factor(Model, levels = c("1-class", "1-class ensemble", "Binary", "Binary ensemble", 
                                     "Full multi-class", "Other multi-class", "Generalised multi-class"))
  )

selected_results_long <- selected_results %>%
  reshape2::melt(
    id.vars = c("Dataset", "Model", "Activity", "training_set", "FalsePositives"),
    measure.vars = c("Adj_F1", "Adj_Precision", "Adj_Recall", "Adj_Accuracy"),
    variable.name = "Metric",
    value.name = "Value"
  ) %>%
  mutate(is_macro = Activity == "Macroaverage") # so I can make it a different shape


# Plot one: dot plot of performance ---------------------------------------
colours <- c(
  # main colours
  "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
  
  # colours maintaining the theme for remaining activities
  "#E67373", "#FFB366", "#B19CD9", "#66B3CC", "#66CC99", "#4D4D99", 
  "#FF8080", "#FFD480", "#9966CC", "#4DB8FF", "#70DB93", "#2D2D86", 
  "#FF9999", "#FFE699", "#8A2BE2", "#00CED1", "#98FB98", "#191970", 
  "#FF69B4", "#F0E68C", "#9370DB", "#20B2AA", "#90EE90", "#000080", 
  "#DDA0DD", "#FFE66C", "#8B008B", "#48D1CC", "#32CD32", "#483D8B"  
)

metric_labels <- c(
  "Adj_F1" = "Adjusted F1 Score",
  "Adj_Precision" = "Adjusted Precision",
  "Adj_Recall" = "Adjusted Recall",
  "Adj_Accuracy" = "Adjusted Accuracy"
)

all_metrics_plot <- ggplot(selected_results_long, 
                           aes(x = Model, y = Value,
                               colour = Activity,
                               shape = is_macro)) +
  geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
  scale_color_manual(values = colours) +
  scale_shape_manual(values = c(16, 8), guide = "none") +
  scale_y_continuous(limits = c(-0.1, 1), breaks = seq(0, 1, by = 0.2)) +
  facet_grid(Metric ~ training_set,  # Changed to Metric ~ training_set for rows ~ columns layout
             labeller = labeller(Metric = as_labeller(metric_labels))) +
  theme_minimal() +
  labs(x = "Model Type", 
       y = "Score") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 12),
    strip.background = element_blank(),
    panel.spacing = unit(2, "lines")
  )

pdf(
  file = file.path(base_path, "Output", "Plots", ML_method, paste0(dataset_name, "_", training_set, "_all_models_plot", suffix, ".pdf")),
  width = 12,
  height = 8
)
print(full_plot)
dev.off()


# Plot two: False Positives -----------------------------------------------
false_positives_plot <- ggplot(selected_results_long, 
                           aes(x = Model, y = FalsePositives,
                               colour = Activity,
                               shape = is_macro)) +
  geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
  scale_color_manual(values = colours) +
  scale_shape_manual(values = c(16, 8), guide = "none") +
  # scale_y_continuous(limits = c(-0.1, ), breaks = seq(0, 1, by = 0.2)) +
  facet_grid(. ~ training_set) +
  theme_minimal() +
  labs(x = "Model Type", 
       y = "False Positives") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 12),
    strip.background = element_blank(),
    panel.spacing = unit(2, "lines")
  )

pdf(
  file = file.path(base_path, "Output", "Plots", ML_method, paste0(dataset_name, "_", training_set, "_all_models_plot", suffix, ".pdf")),
  width = 12,
  height = 8
)
print(full_plot)
dev.off()


# Plot three: confusion matrices ------------------------------------------
confusion_data <- fread(file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", ML_method, "_complete_confusion.csv")))
confusion_data <- confusion_data %>% 
  mutate(Correct = ifelse(ground_truth_labels == prediction_labels, TRUE, FALSE)) %>%
  mutate(Correct == ifelse(prediction_labels == "Other" & !ground_truth_labels %in% target_activities, TRUE, Correct)) %>%
  mutate(Correct = as.factor(Correct)) %>%
  filter(!N == 0)

# select some to look at
one_set <- confusion_data %>% filter(training_set == "target", model == "OCC")

# make each row its own observation
expanded_confusion <- one_set %>%
  # Use uncount to create one row per observation
  uncount(N) %>%
  # Add a unique ID for each observation within each ground_truth/prediction combination
  group_by(ground_truth_labels, prediction_labels) %>%
  mutate(obs_id = row_number()) %>%
  ungroup()

# Create the plot
confusion_dots_plot2 <- ggplot(confusion_data, 
                              aes(x = ground_truth_labels, 
                                  y = prediction_labels,
                                  colour = Correct)) +
  # Add jittered points for each observation
  geom_jitter(size = 0.5, 
              alpha = 0.8,
              width = 0.2,  # Adjust these values to control spread
              height = 0.2) +
  # Add gridlines to make the grid structure clear
  geom_hline(yintercept = sequence(length(all_levels)), 
             color = "grey90") +
  geom_vline(xintercept = sequence(length(all_levels)), 
             color = "grey90") +
  # Make the plot square
  coord_fixed() +
  # Customize theme
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(hjust = 1),
        panel.grid = element_blank()) +
  labs(x = "Ground Truth",
       y = "Predictions") +
  facet_grid(model ~ training_set)




# Other / Old -------------------------------------------------------------

# generate plot of perofrmance metrics
absolute_performance(combined_results, dataset_name, base_path, training_set = training_set)
zero_performance(combined_results, dataset_name, base_path, training_set)
# random_performance(combined_results, dataset_name, base_path) # need to fix



all_performance_plots(combined_results, dataset_name, base_path)










# add in the ground truth labels
training_data <- fread(
  file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv"))) %>%
  as.data.table()
training_data$Activity <- str_to_title(training_data$Activity)



# tuning plot -------------------------------------------------------------

# time for tuning per model
tuning_files <- list.files(file.path(base_path, "Output", "Tuning"), pattern = "*.csv", full.names = TRUE)
combined_tuning_data <- lapply(tuning_files, read.csv) %>%
  bind_rows()
combined_tuning_data$behaviour_or_activity <- str_to_title(combined_tuning_data$behaviour_or_activity) # format for consistency



colours <- c(
  "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
  "#FF8C42", "#4F7190", "#BF6D4C", "#7CB518", "#6A4C93"
)

combined_tuning_plot <- combined_tuning_data %>% 
  mutate(
    model_kind = ifelse(model_type == "Multi", behaviour_or_activity, "Individual Models"),
    model_kind = factor(
      model_kind, 
      levels = c("Activity", "Otheractivity", "Generalisedactivity", "Individual Models")
    ),
    model_type = factor(
      model_type, 
      levels = c("OCC", "Binary", "Multi"),
      labels = c("1-class", "Binary", "Multi")
    )
  )

tuning_time <- ggplot(combined_tuning_plot, aes(x = model_type, y = elapsed, colour = model_kind, shape = data_name)) +
  geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
  scale_y_log10() +
  scale_color_manual(values = colours) +
  scale_shape_manual(values= c(16, 17)) +
  guides(
    color = guide_legend(title = "Activity"),
    shape = guide_legend(title = "Dataset")
  ) +
  theme_minimal() +
  labs(x = "Model Type",
       y = "Elapsed Time (sec) (log scale") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 12),
    strip.background = element_blank(),
    panel.spacing = unit(2, "lines")
  )

ggsave(
  filename = file.path(base_path, "Output", "Plots", "Tuning_elapsed_time.pdf"),
  plot = tuning_time,
  width = 12,
  height = 8,
  device = cairo_pdf  # Use cairo_pdf for better handling of transparency
)




duration <- combined_tuning_plot %>% select(1:6)
fwrite(duration, file.path(base_path, "Output", "Tuning", "Tuning_durations.csv"))

