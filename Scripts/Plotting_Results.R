
# Plotting model performance ----------------------------------------------

results <- fread(file.path(base_path, "Output", "Test_Performance.csv"))

results_long <- melt(results, id.vars = c("data", "model_type", "behaviour"), 
                     measure.vars = c("f1", "precision", "recall", "accuracy"), 
                     variable.name = "metric", value.name = "value")

dog_data <- results_long %>%
  filter(data == dataset_name) %>%
  filter(!behaviour == "Other") %>%
  mutate(
    behaviour = factor(behaviour, levels = c("Macro", "Walking", "Eating", "Lying Chest", "Shaking")),
    model_type = factor(model_type, levels = c("OCC", "Activity", "OtherActivity", "GeneralisedActivity"))
  )




colours <- c("#F3E37C", "#C1CAD6", "#037971", "#F7996E", "#AD343E")
colours2 <- c("#EDAE49", "#8F2D56", "#5D737E", "#DE6449")
shapes <- c(15, 16, 17, 18, 20)

# Create the plot: simple
ggplot(dog_data, aes(x = model_type, y = value, color = metric, shape = metric)) +
  geom_point( size = 5) + #position = position_dodge(width = 0.5),
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
    axis.title.y = element_blank(),  # Remove y-axis label for all facets
    panel.spacing = unit(2, "lines")  # Adjust space between panels
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
  facet_grid(metric ~ behaviour, switch = "y") +  # Use facet_grid and switch to move metric labels to the left
  theme_minimal() +
  labs(x = "Model Type") +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),  # Add black border
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
    strip.text.x = element_text(size = 12),  # Keep behavior labels on top
    strip.text.y = element_text(size = 12),  # Metric labels on the left
    strip.placement = "outside",  # Place facet labels outside the plot
    strip.background = element_blank(),  # Remove facet label background
    axis.title.y = element_blank(),  # Remove y-axis label for all facets
    panel.spacing = unit(2, "lines")  # Adjust space between panels
  )

# plot 3: change the axes
dog_data_reduced <- dog_data %>% filter(model_type %in% c("OCC", "Activity"), 
                                        !metric == "accuracy")

ggplot(dog_data_reduced, aes(x = model_type, y = value, colour = behaviour, shape = behaviour)) +
  geom_point(size = 5) +
  scale_color_manual(values = colours) +
  scale_shape_manual(values = shapes) +
  scale_y_continuous(limits = c(0, 1)) +
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
    axis.title.y = element_blank(),  # Remove y-axis label for all facets
    panel.spacing = unit(2, "lines")  # Adjust space between panels
  )

# plot 4: Show the variance

    


