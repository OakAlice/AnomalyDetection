
# Plotting the performance comparing dichotomous with multiclass ----------

# read in
combined_results <- fread(file = file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", ML_method, "_complete_test_performance.csv"))) %>% as.data.frame()

combined_results <- combined_results %>% mutate(Activity = ifelse(Activity == "Weightedmacroaverage", "Macroaverage", Activity))


# generate plot of perofrmance metrics
absolute_performance(combined_results, dataset_name, base_path)
zero_performance(combined_results, dataset_name, base_path)
# random_performance(combined_results, dataset_name, base_path) # need to fix

all_f1_plots(combined_results, dataset_name, base_path)

full_multi(combined_results, dataset_name, base_path)

performance_to_volume(combined_results, training_data, dataset_name, base_path)













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

