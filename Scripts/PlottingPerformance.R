
# Plotting the performance comparing dichotomous with multiclass ----------

# read in
combined_results <- fread(file = file.path(base_path, "Output", "Testing", "Complete_test_performance.csv"))
combined_results$Activity <- str_to_title(combined_results$Activity) # format for consistency

dog_results <- combined_results %>% filter(Dataset == "Vehkaoja_Dog")
seal_results <- combined_results %>% filter(Dataset == "Ladds_Seal")

# generate plot of perofrmance metrics
generate_plots(seal_results, "Ladds_Seal", base_path)
generate_plots(dog_results, "Vehkaoja_Dog", base_path)



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
