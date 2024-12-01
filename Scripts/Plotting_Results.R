
# Plotting model performance ----------------------------------------------

# Load and process model performance results
results <- fread(file = file.path(base_path, "Output", "Test_Performance.csv"))

# Reshape the results for plotting
results <- results %>%
  mutate(across(c(f1, precision, recall, accuracy), as.numeric))

results_long <- melt(
  results,
  id.vars = c("data", "model_type", "behaviour"),
  measure.vars = c("f1", "precision", "recall", "accuracy"),
  variable.name = "metric",
  value.name = "value"
)

# Define Dog Data
dog_data <- results_long %>%
  filter(data == "Vehkaoja_Dog") %>%
  mutate(
    behaviour = factor(behaviour, levels = c("Macro", "Walking", "Eating", "Lying Chest", "Shaking", "Other")),
    model_type = factor(model_type, levels = c("OCC", "Activity", "OtherActivity", "GeneralisedActivity")),
    is_na = is.na(value)
  )

dog_data_reduced <- dog_data %>%
  filter(
    model_type %in% c("OCC", "Activity"),
    metric != "accuracy"
  )

# Define Seal Data
seal_data <- results_long %>%
  filter(data == "Ladds_Seal") %>%
  mutate(
    behaviour = factor(behaviour, levels = c("Macro", "Swimming", "Chewing", "Still", "Facerub", "Other")),
    model_type = factor(model_type, levels = c("OCC", "Activity", "OtherActivity", "GeneralisedActivity")),
    is_na = is.na(value)
  )

seal_data_reduced <- seal_data %>%
  filter(
    model_type %in% c("OCC", "Activity"),
    metric != "accuracy"
  )

# Define colors and shapes
colours <- c("#A63A50", "#FFCF56", "#D4B2D8", "#3891A6", "#3BB273", "#031D44")
shapes <- c(15, 16, 17, 18, 20, 8)
# shapes <- c(0, 1, 2, 5, 6, 11)

# Choose which data to use
animal_data <- seal_data  # Change to `seal_data` if needed

# Define the plot
ggplot(animal_data, aes(x = model_type, y = value, colour = behaviour, shape = behaviour)) +
  geom_point(size = 5, na.rm = TRUE, alpha = 0.8) +  # Regular points
  geom_point(
    aes(x = model_type, y = -0.1, colour = behaviour, shape = behaviour),  # Offset NA points slightly below 0
    data = animal_data %>% filter(is_na),
    size = 5,
    alpha = 0.8  # Fixed transparency for NA points
  ) +
  scale_color_manual(values = colours) +
  scale_shape_manual(values = shapes) +
  scale_y_continuous(
    limits = c(-0.1, 1),
    breaks = seq(0, 1, by = 0.2)
  ) +  # Extend limits to include NA row
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





# Other stuff -------------------------------------------------------------


# Create the plot: simple
ggplot(seal_data, aes(x = model_type, y = value, color = metric, shape = metric)) +
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


