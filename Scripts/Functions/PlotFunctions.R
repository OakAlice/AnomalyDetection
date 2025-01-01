# Plotting model performance ----------------------------------------------
generate_plots <- function(combined_results, dataset_name, base_path) {
 
  # Reshape the results
  results_long <- combined_results %>%
    mutate(across(c(F1_Score, Precision, Recall, Accuracy), as.numeric)) %>%
    melt(
      id.vars = c("Dataset", "Model", "Activity"),
      measure.vars = c("F1_Score", "Precision", "Recall", "Accuracy"),
      variable.name = "Metric",
      value.name = "Value"
    )
  
  # rename the model types
  results_long <- results_long %>%
    mutate(Model = case_when(
      Model == "OCC" ~ "1-class",
      Model == "Binary" ~ "Binary",
      Model == "Activity" ~ "Full multi-class",
      Model == "OtherActivity" ~ "Other multi-class",
      Model == "GeneralisedActivity" ~ "Generalised multi-class",
      TRUE ~ Model  # Preserve any values not listed above
    ))
  
  # Define processing logic based on dataset
  if (dataset_name == "Vehkaoja_Dog") {
    animal_data <- results_long %>%
      filter(
        data == dataset_name,
        metric != "accuracy"
      ) %>%
      mutate(
        behaviour = factor(behaviour, levels = c("Macro", "Walking", "Eating", "Lying Chest", "Shaking", "Other")),
        model_type = factor(model_type, levels = c("1-class", "Binary", "Full multi-class", "Other multi-class", "Generalised multi-class")),
        is_na = is.na(value)
      )
    
    animal_data_reduced <- animal_data %>%
      filter(model_type %in% c("1-class", "Full multi-class"))
    
  } else if (dataset_name == "Ladds_Seal") {
    animal_data <- results_long %>%
      filter(
        Dataset == dataset_name,
        Metric != "Accuracy"
      ) %>%
      mutate(Activity = str_to_title(Activity)) %>%
      mutate(
        Activity = factor(Activity, levels = c("MacroAverage", "Swimming", "Chewing", "Still", "Facerub", "Other")),
        Model = factor(Model, levels = c("1-class", "Binary", "Full multi-class", "Other multi-class", "Generalised multi-class")),
        is_na = is.na(Value)
      )
    
    animal_data_reduced <- animal_data %>%
      filter(Model %in% c("1-class", "Full multi-class"))
    
  } else {
    stop("Dont know how to work with this data - have to define in function")
  }
  
  # Define colors and shapes
  colours <- c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
  shapes <- c(15, 16, 17, 18, 20, 8)
  
  # Helper function to create plots
  create_plot <- function(data) {
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
  
  # Generate plots
  full_plot <- create_plot(animal_data)
  reduced_plot <- create_plot(animal_data_reduced)
  
  # Save plots as PDFs
  pdf(
    file = file.path(base_path, "Output", "Plots", paste0(dataset_name, "_all_models_plot.pdf")),
    width = 12,
    height = 8
  )
  print(full_plot)
  dev.off()
  
  pdf(
    file = file.path(base_path, "Output", "Plots", paste0(dataset_name, "_reduced_models_plot.pdf")),
    width = 10,
    height = 6
  )
  print(reduced_plot)
  dev.off()
  
  message("Plots saved successfully.")
}
