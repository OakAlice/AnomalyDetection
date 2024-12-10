# Plotting model performance ----------------------------------------------
generate_plots <- function(base_path, dataset_name, results_file_name) {
  # Load and process model performance results # change the name of the file here
  results <- fread(file = file.path(base_path, "Output", results_file_name))

  # Reshape the results
  results_long <- results %>%
    mutate(across(c(f1, precision, recall, accuracy), as.numeric)) %>%
    melt(
      id.vars = c("data", "model_type", "behaviour"),
      measure.vars = c("f1", "precision", "recall", "accuracy"),
      variable.name = "metric",
      value.name = "value"
    )

  # Define processing logic based on dataset
  if (dataset_name == "Vehkaoja_Dog") {
    animal_data <- results_long %>%
      filter(
        data == dataset_name,
        metric != "accuracy"
      ) %>%
      mutate(
        behaviour = factor(behaviour, levels = c("Macro", "Walking", "Eating", "Lying Chest", "Shaking", "Other")),
        model_type = factor(model_type, levels = c("1-class", "Full multi-class", "Other multi-class", "Generalised multi-class")),
        is_na = is.na(value)
      )

    animal_data_reduced <- animal_data %>%
      filter(model_type %in% c("1-class", "Full multi-class"))
  } else if (dataset_name == "Ladds_Seal") {
    animal_data <- results_long %>%
      filter(
        data == dataset_name,
        metric != "accuracy"
      ) %>%
      mutate(
        behaviour = factor(behaviour, levels = c("Macro", "Swimming", "Chewing", "Still", "Facerub", "Other")),
        model_type = factor(model_type, levels = c("1-class", "Full multi-class", "Other multi-class", "Generalised multi-class")),
        is_na = is.na(value)
      )

    animal_data_reduced <- animal_data %>%
      filter(model_type %in% c("1-class", "Full multi-class"))
  } else {
    stop("Dont know how to level this data - have to define in function")
  }

  # Define colors and shapes
  colours <- c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
  shapes <- c(15, 16, 17, 18, 20, 8)

  # Helper function to create plots
  create_plot <- function(data) {
    ggplot(data, aes(x = model_type, y = value, colour = behaviour, shape = behaviour)) +
      geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +
      geom_point(
        aes(x = model_type, y = -0.1, colour = behaviour, shape = behaviour),
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
        color = guide_legend(title = "Behaviour"),
        shape = guide_legend(title = "Behaviour")
      ) +
      facet_wrap(. ~ metric, ncol = 5) +
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




generate_plots(base_path, "Ladds_Seal", "Test_Performance2.csv")
generate_plots(base_path, "Vehkaoja_Dog", "Test_Performance2.csv")







# Other stuff -------------------------------------------------------------


# Create the plot: simple
ggplot(seal_data, aes(x = model_type, y = value, color = metric, shape = metric)) +
  geom_point(size = 5) + # position = position_dodge(width = 0.5),
  scale_color_manual(values = colours) +
  scale_shape_manual(values = shapes) +
  scale_y_continuous(limits = c(0, 1)) +
  guides(
    color = guide_legend(title = "Metric"),
    shape = guide_legend(title = "Metric")
  ) +
  facet_wrap(. ~ behaviour, ncol = 5) +
  theme_minimal() +
  labs(x = "Model Type") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 12),
    strip.background = element_blank(),
    axis.title.y = element_blank(), # Remove y-axis label for all facets
    panel.spacing = unit(2, "lines") # Adjust space between panels
  )

# plot 2: extra grid
ggplot(dog_data, aes(x = model_type, y = value, color = metric, shape = metric)) +
  geom_point(size = 5) + # position = position_dodge(width = 0.5),
  scale_color_manual(values = colours) +
  scale_shape_manual(values = shapes) +
  scale_y_continuous(limits = c(0, 1)) +
  guides(
    color = guide_legend(title = "Metric"),
    shape = guide_legend(title = "Metric")
  ) +
  facet_grid(metric ~ behaviour, switch = "y") + # Use facet_grid and switch to move metric labels to the left
  theme_minimal() +
  labs(x = "Model Type") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1), # Add black border
    axis.text.x = element_text(angle = 45, hjust = 1), # Rotate x-axis labels
    strip.text.x = element_text(size = 12), # Keep behavior labels on top
    strip.text.y = element_text(size = 12), # Metric labels on the left
    strip.placement = "outside", # Place facet labels outside the plot
    strip.background = element_blank(), # Remove facet label background
    axis.title.y = element_blank(), # Remove y-axis label for all facets
    panel.spacing = unit(2, "lines") # Adjust space between panels
  )
