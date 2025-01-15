# Plotting model performance ----------------------------------------------
# General helping functions -----------------------------------------------
create_base_plot <- function(data, colours, shapes) {
  ggplot(data, aes(x = Model, y = Value, colour = Activity, shape = Activity)) +
    geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
    geom_point(
      aes(x = Model, y = -0.1, colour = Activity, shape = Activity),
      data = data %>% filter(is_na),
      size = 5,
      alpha = 0.8
    ) +
    scale_color_manual(values = colours) +
    scale_shape_manual(values = shapes) +
    scale_y_continuous(
      limits = c(-0.1, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    guides(
      color = guide_legend(title = "Activity"),
      shape = guide_legend(title = "Activity")
    ) +
    facet_wrap(. ~ Metric, ncol = 5) +
    theme_minimal() +
    labs(x = "Model Type") +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(size = 12),
      strip.background = element_blank(),
      axis.title.y = element_blank(),
      panel.spacing = unit(2, "lines")
    )
}

process_dataset <- function(results_long, dataset_name) {
  activity_levels <- if(dataset_name == "Vehkaoja_Dog") {
    c("Macroaverage", "Walking", "Eating", "Lying Chest", "Shaking", "Other")
  } else if(dataset_name == "Ladds_Seal") {
    c("Macroaverage", "Swimming", "Chewing", "Still", "Facerub", "Other") 
  } else {
    stop("Unknown dataset name")
  }
  
  animal_data <- results_long %>%
    filter(Dataset == dataset_name) %>%
    mutate(Activity = if(dataset_name == "Ladds_Seal") str_to_title(Activity) else Activity) %>%
    mutate(
      Activity = factor(Activity, levels = activity_levels),
      Model = factor(Model, levels = c("1-class", "1-class ensemble", "Binary", "Binary ensemble", 
                                     "Full multi-class", "Other multi-class", "Generalised multi-class")),
      is_na = is.na(Value)
    ) %>%
    filter(Activity %in% activity_levels)
  
  list(
    full = animal_data,
    reduced = animal_data %>% filter(Model %in% c("Binary", "Binary ensemble", "Full multi-class"))
  )
}

save_plots <- function(full_plot, reduced_plot, dataset_name, base_path, suffix = "") {
  pdf(
    file = file.path(base_path, "Output", "Plots", ML_method, paste0(dataset_name, "_all_models_plot", suffix, ".pdf")),
    width = 12,
    height = 8
  )
  print(full_plot)
  dev.off()
  
  # pdf(
  #   file = file.path(base_path, "Output", "Plots", paste0(dataset_name, "_reduced_models_plot", suffix, ".pdf")),
  #   width = 10,
  #   height = 6
  # )
  # print(reduced_plot)
  # dev.off()
  
  message("Plots saved successfully.")
}

plot_and_save <- function(results_long, dataset_name, base_path, suffix = "") {
  colours <- c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
  shapes <- c(15, 16, 17, 18, 20, 8)
  
  processed_data <- process_dataset(results_long, dataset_name)
  
  full_plot <- create_base_plot(processed_data$full, colours, shapes)
  reduced_plot <- create_base_plot(processed_data$reduced, colours, shapes)
  
  save_plots(full_plot, reduced_plot, dataset_name, base_path, suffix)
}

prepare_results <- function(combined_results) {
  combined_results %>%
    reshape2::melt(
      id.vars = c("Dataset", "Model", "Activity"),
      measure.vars = c("F1_Score", "Precision", "Recall", "Accuracy"),
      variable.name = "Metric",
      value.name = "Value"
    ) %>%
    mutate(Model = case_when(
      Model == "OCC" ~ "1-class",
      Model == "Binary" ~ "Binary", 
      Model == "OCC_Ensemble" ~ "1-class ensemble",
      Model == "Binary_Ensemble" ~ "Binary ensemble",
      Model == "Activity" ~ "Full multi-class",
      Model == "OtherActivity" ~ "Other multi-class",
      Model == "GeneralisedActivity" ~ "Generalised multi-class",
      TRUE ~ Model
    ))
}


# Main plotting functions -----------------------------------------------
# absolute
absolute_performance <- function(combined_results, dataset_name, base_path) {
  results_long <- prepare_results(combined_results)
  plot_and_save(results_long, dataset_name, base_path)
}

# adjusted for the zero rate performance
zero_performance <- function(combined_results, dataset_name, base_path) {
  
  combined_results_zero <- combined_results %>%
    select(Dataset, Model, Activity, Zero_adj_F1_Score, Zero_adj_Precision, Zero_adj_Recall, Zero_adj_Accuracy) %>%
    rename(F1_Score = Zero_adj_F1_Score,
           Precision = Zero_adj_Precision,
           Recall = Zero_adj_Recall,
           Accuracy = Zero_adj_Accuracy) %>%
    mutate(across(where(is.numeric), ~replace(., is.nan(.), 0))) 
   
  results_long <- prepare_results(combined_results_zero)
  
  
  plot_and_save(results_long, dataset_name, base_path, "_zero")
}



# adjusted for random guessing performance
random_performance <- function(combined_results, dataset_name, base_path) {
  
  combined_results_rand <- combined_results %>%
    select(Dataset, Model, Activity, Rand_adj_F1_Score, Rand_adj_Precision, Rand_adj_Recall, Rand_adj_Accuracy) %>%
    rename(F1_Score = Rand_adj_F1_Score,
           Precision = Rand_adj_Precision,
           Recall = Rand_adj_Recall,
           Accuracy = Rand_adj_Accuracy) %>%
    mutate(across(where(is.numeric), ~replace(., is.nan(.), 0))) 
  
  results_long <- prepare_results(combined_results_rand)
  
  plot_and_save(results_long, dataset_name, base_path, "_random")
}



all_f1_plots <- function(combined_results, dataset_name, base_path) {
  # Prepare the three different F1 score datasets
  unadjusted <- combined_results %>%
    select(Dataset, Model, Activity, F1_Score, Accuracy) %>%
    mutate(Adjustment = "Unadjusted")
  
  random_adjusted_prev <- combined_results %>%
    select(Dataset, Model, Activity, Rand_adj_F1_Score_prev, Rand_adj_Accuracy_prev) %>%
    rename(F1_Score = Rand_adj_F1_Score_prev,
           Accuracy = Rand_adj_Accuracy_prev) %>%
    mutate(Adjustment = "Random-adjusted-prev")
  
  random_adjusted_equal <- combined_results %>%
    select(Dataset, Model, Activity, Rand_adj_F1_Score_equal, Rand_adj_Accuracy_equal) %>%
    rename(F1_Score = Rand_adj_F1_Score_equal,
           Accuracy = Rand_adj_Accuracy_equal) %>%
    mutate(Adjustment = "Random-adjusted-equal")
  
  zero_adjusted <- combined_results %>%
    select(Dataset, Model, Activity, Zero_adj_F1_Score, Zero_adj_Accuracy) %>%
    rename(F1_Score = Zero_adj_F1_Score,
           Accuracy = Zero_adj_Accuracy) %>%
    mutate(Adjustment = "Zero-adjusted")
  
  # Combine all datasets
  behaviours <- unlist(unique(unadjusted$Activity[unadjusted$Model == "OtherActivity"]))
  activity_levels <- if(dataset_name == "Vehkaoja_Dog") {
    c("Macroaverage", "Walking", "Eating", "Lying Chest", "Shaking", "Other")
  } else if(dataset_name == "Ladds_Seal") {
    c("Macroaverage", "Swimming", "Chewing", "Still", "Facerub", "Other") 
  } else {
    stop("Unknown dataset name")
  }
  
  all_data <- bind_rows(unadjusted, random_adjusted_prev, random_adjusted_equal, zero_adjusted) %>%
    filter(Activity %in% c(behaviours)) %>%
    mutate(Model = case_when(
      Model == "OCC" ~ "1-class",
      Model == "Binary" ~ "Binary", 
      Model == "OCC_Ensemble" ~ "1-class ensemble",
      Model == "Binary_Ensemble" ~ "Binary ensemble",
      Model == "Activity" ~ "Full multi-class",
      Model == "OtherActivity" ~ "Other multi-class",
      Model == "GeneralisedActivity" ~ "Generalised multi-class",
      TRUE ~ Model
    )) %>%
    mutate(
      Activity = factor(Activity, levels = activity_levels),
      Model = factor(Model, levels = c("1-class", "1-class ensemble", "Binary", "Binary ensemble", 
                                     "Full multi-class", "Other multi-class", "Generalised multi-class")),
      Adjustment = factor(Adjustment, levels = c("Unadjusted", "Zero-adjusted", "Random-adjusted-prev", "Random-adjusted-equal")),
      is_na = is.na(F1_Score)
    ) %>%
    filter(Dataset == dataset_name)

  
  # Define colors and shapes
  colours <- c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
  shapes <- c(15, 16, 17, 18, 20, 8)
  
  # Create plot
  full_f1_plot <- ggplot(all_data, aes(x = Model, y = F1_Score, colour = Activity, shape = Activity)) +
    geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
    geom_point(
      data = all_data %>% filter(is_na),
      aes(y = -0.1),
      size = 5,
      alpha = 0.8
    ) +
    scale_color_manual(values = colours) +
    scale_shape_manual(values = shapes) +
    scale_y_continuous(
      limits = c(-0.1, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    facet_wrap(. ~ Adjustment, ncol = 4) +
    theme_minimal() +
    labs(
      x = "Model Type",
      y = "F1 Score"
    ) +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(size = 12),
      strip.background = element_blank(),
      panel.spacing = unit(2, "lines"),
      plot.title = element_text(hjust = 0.5)
    )
  
  full_accuracy_plot <- ggplot(all_data, aes(x = Model, y = Accuracy, colour = Activity, shape = Activity)) +
    geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
    geom_point(
      data = all_data %>% filter(is_na),
      aes(y = -0.1),
      size = 5,
      alpha = 0.8
    ) +
    scale_color_manual(values = colours) +
    scale_shape_manual(values = shapes) +
    scale_y_continuous(
      limits = c(-0.1, 1),
      breaks = seq(0, 1, by = 0.2)
    ) +
    facet_wrap(. ~ Adjustment, ncol = 4) +
    theme_minimal() +
    labs(
      x = "Model Type",
      y = "F1 Score"
    ) +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(size = 12),
      strip.background = element_blank(),
      panel.spacing = unit(2, "lines"),
      plot.title = element_text(hjust = 0.5)
    )
  
  # Save plot
  pdf(
    file = file.path(base_path, "Output", "Plots", ML_method, paste0(dataset_name, "_all_f1_plot.pdf")),
    width = 12,
    height = 8
  )
  print(full_f1_plot)
  dev.off()
  
  pdf(
    file = file.path(base_path, "Output", "Plots", ML_method, paste0(dataset_name, "_all_accuracy_plot.pdf")),
    width = 12,
    height = 8
  )
  print(full_accuracy_plot)
  dev.off()
  
  message("F1 score and accuracy comparison plots saved successfully.")
  
  # Return the plot invisibly
  invisible(full_plot)
}





# plot all the bbehaviours from the full multiclass model
full_multi <- function(combined_results, dataset_name, base_path) {
  # Filter and select relevant data
  combined_results <- combined_results %>%
    select(Dataset, Model, Activity, F1_Score, Precision, Recall, Accuracy) %>%
    filter(Model %in% c("Activity", "Binary"))
  
  # Reshape to long format
  results_long <- combined_results %>%
    mutate(across(c(F1_Score, Precision, Recall, Accuracy), as.numeric)) %>%
    reshape2::melt(
      id.vars = c("Dataset", "Model", "Activity"),
      measure.vars = c("F1_Score", "Precision", "Recall", "Accuracy"),
      variable.name = "Metric",
      value.name = "Value"
    ) %>%
    mutate(Model = case_when(
      Model == "Binary" ~ "Binary",
      Model == "Activity" ~ "Full multi-class"
    )) %>%
    mutate(
      Activity = factor(Activity, levels = unique(Activity)),
      Model = factor(Model, levels = c("Binary", "Full multi-class")),
      is_na = is.na(Value)
    )
  
  # Define color palette
  colours <- c(
    "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
    "#8B523E", "#C39E69", "#7F7059", "#5C7A4A", "#4F7190", "#BF6D4C",
    "#9B7551", "#A29C7A", "#72966D", "#5A82A1", "#A86C63", "#D2B37E",
    "#66866C", "#4E6A72", "#9B705B", "#B29C6A", "#729F74", "#4F5665",
    "#8A554A", "#BFA57A", "#788666", "#617884", "#D08E45", "#E6BF6D",
    "#C4874A", "#A6733B", "#D9A85B", "#F4C879"
  )
  
  # Create plot
  plot <- ggplot(results_long, aes(x = Model, y = Value, colour = Activity)) +
    geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
    geom_point(
      data = results_long %>% filter(is_na),
      aes(y = -1),
      size = 5,
      alpha = 0.8
    ) +
    scale_color_manual(values = colours) +
    scale_y_continuous(limits = c(-0.1, 1), breaks = seq(0, 1, by = 0.2)) +
    facet_wrap(. ~ Metric, ncol = 5) +
    theme_minimal() +
    labs(x = "Model Type") +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(size = 12),
      strip.background = element_blank(),
      axis.title.y = element_blank(),
      panel.spacing = unit(2, "lines")
    )
  
  # Save plot
  pdf(
    file = file.path(base_path, "Output", "Plots", paste0(dataset_name, "_full_activity.pdf")),
    width = 12,
    height = 8
  )
  print(plot)
  dev.off()
  
  message("Plots saved successfully.")
}






performance_to_volume <- function(combined_results, training_data, dataset_name, base_path) {
  # Calculate activity volumes
  volumes <- list(
    Activity = training_data %>% count(Activity),
    GeneralisedActivity = training_data %>% count(GeneralisedActivity)
  )
  
  # Prepare results data
  plot_data <- combined_results %>%
    select(Dataset, Model, Activity, F1_Score, Accuracy, Random_F1, Random_Accuracy) %>%
    merge(volumes$Activity, by = "Activity") %>%
    filter(Model == "Activity", n > 0)  # Ensure positive n for log scale
  
  # Define color palette
  colours26 <- c(
    "#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44",
    "#8B523E", "#C39E69", "#7F7059", "#5C7A4A", "#4F7190", "#BF6D4C",
    "#9B7551", "#A29C7A", "#72966D", "#5A82A1", "#A86C63", "#D2B37E",
    "#66866C", "#4E6A72", "#9B705B", "#B29C6A", "#729F74", "#4F5665",
    "#8A554A", "#BFA57A", "#788666", "#617884", "#D08E45", "#E6BF6D"
  )
  
  # Calculate R-squared value
  lm_model <- lm(Random_F1 ~ n, data = plot_data)
  r2_value <- round(summary(lm_model)$r.squared, 3)
  
  # Create base theme
  base_theme <- theme_minimal() +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      axis.text.x = element_text(angle = 45, hjust = 1),
      strip.text = element_text(size = 12),
      strip.background = element_blank(),
      axis.title.y = element_text(size = 12),
      panel.spacing = unit(2, "lines")
    )
  
  # Create standard scale plot
  p1 <- ggplot(plot_data, aes(x = n, y = Random_F1, colour = Activity)) +
    geom_point(size = 5, alpha = 0.8) +
    scale_color_manual(values = colours26) +
    geom_smooth(method = "lm", se = FALSE, color = "grey50", linetype = "dashed", alpha = 0.8) +
    labs(
      x = "Number of Samples",
      y = "F1 Score (Adjusted for random guessing F1)"
    ) +
    annotate(
      "text",
      x = max(plot_data$n) * 0.8,
      y = min(plot_data$Random_F1) + (max(plot_data$Random_F1) - min(plot_data$Random_F1)) * 0.9,
      label = sprintf("R² = %.3f", r2_value),
      size = 4,
      fontface = "bold"
    ) +
    base_theme +
    theme(legend.position = "none")  # Remove legend from first plot
  
  # Create log scale plot
  p2 <- ggplot(plot_data, aes(x = n, y = Random_F1, colour = Activity)) +
    geom_point(size = 5, alpha = 0.8) +
    scale_color_manual(values = colours26) +
    scale_x_log10() +
    labs(
      x = "Number of Samples (log scale)",
      y = "F1 Score (Adjusted for random guessing F1)",
      color = "Activity"
    ) +
    base_theme
  
  # Combine plots using patchwork
  combined_plot <- p1 + p2 + 
    plot_layout(guides = "collect") &  # Collect legends
    theme(legend.position = "right")   # Position the combined legend
  
  # Ensure output directory exists
  output_dir <- file.path(base_path, "Output", "Plots")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save to PDF
  ggsave(
    filename = file.path(output_dir, paste0(dataset_name, "_performance_volume_log.pdf")),
    plot = combined_plot,
    width = 12,
    height = 8,
    device = cairo_pdf  # Use cairo_pdf for better handling of transparency
  )
  
  # Return the combined plot invisibly
  invisible(combined_plot)
}
