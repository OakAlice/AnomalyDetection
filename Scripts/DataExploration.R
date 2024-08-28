# Plotting the features

# plot the training_data features ####
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

data <- data_other %>%
  group_by(Activity) %>%
  #slice_head(n = 10000) %>%
  ungroup()
features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "auto", "zero", "fft")
data_processed <- process_data(na.omit(data), features_list, window_length = 1, 
                                        overlap_percent = 0, down_Hz = 100, 
                                        feature_normalisation = "Standardised")

# Reshape data from wide to long format
data_long <- data_processed %>%
  pivot_longer(
    cols = starts_with("mean_") | starts_with("max_") | starts_with("min_") | starts_with("sd_") | starts_with("entropy_") | starts_with("auto_") | starts_with("SMA") | starts_with("minODBA") | starts_with("maxODBA") | starts_with("minVDBA") | starts_with("maxVDBA") | starts_with("cor_"),
    names_to = "Feature",
    values_to = "Value"
  )

# Create the plot
ggplot(data_long, aes(x = Value, fill = Activity)) +
  geom_histogram(position = "identity", alpha = 0.6, binwidth = 0.1) +
  facet_wrap(~ Feature, scales = "free_x", ncol = 4) +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),  # Remove x-axis text
    axis.text.y = element_blank(),  # Remove y-axis text
    axis.title.x = element_blank(), # Remove x-axis title
    axis.title.y = element_blank(), # Remove y-axis title
    axis.ticks = element_blank()    # Remove axis ticks
  ) +
  labs(
    fill = "Activity"
  )


# same idea as above but different view ####
data_long <- data_processed %>%
  pivot_longer(cols = -c(Activity, Timestamp, ID), names_to = "feature", values_to = "value")


data_long_ind <- data_long %>% filter(ID %in% 22)

# Plot using ggplot
ggplot(data_long_ind, aes(x = Activity, y = value, colour = Activity)) +
  geom_jitter() +
  facet_wrap(~ feature, scales = "free_y") +
  theme(axis.text.x = element_blank()) +
  theme_minimal()+
  labs(title = "Jitter Plot of Features by Activity",
       x = "Activity",
       y = "Value")


# interactions between the features ####
library(GGally)

# Select the numeric columns for pairwise plotting
numeric_data <- data_processed %>%
  select(-c(Activity, Timestamp, ID
            )) %>%
  filter_all(all_vars(is.finite(.))) %>%
  na.omit()

# Pairwise scatter plots using ggpairs
numeric_columns <- colnames(numeric_data)

# Create pairwise scatter plots
for (i in 1:(length(numeric_columns) - 1)) {
  for (j in (i + 1):length(numeric_columns)) {
    p <- ggplot(data_processed, aes_string(x = numeric_columns[i], y = numeric_columns[j], colour = "Activity")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(title = paste("Scatter Plot of", numeric_columns[i], "vs", numeric_columns[j]),
           x = numeric_columns[i],
           y = numeric_columns[j])
    
    print(p)
  }
}



## Here ####
library(patchwork)
data_long <- data_processed %>%
  select(-c(Timestamp, ID)) %>%
  gather(key = "feature", value = "value", -Activity)

features <- unique(data_long$feature)

# Initialize an empty plot object
plot_list <- list()

# Iterate through each pair of features
for(i in 1:(length(features)-1)) {
  for(j in (i+1):length(features)) {
    plot <- ggplot(data_processed, aes_string(x = features[i], y = features[j], colour = "Activity")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(x = features[i], y = features[j]) +
      theme(legend.position = "none")
    
    plot_list[[paste(features[i], features[j], sep = "_vs_")]] <- plot
  }
}

# Combine all plots into a grid using patchwork
combined_plot <- wrap_plots(plot_list, ncol = 3) +
  plot_layout(guides = "collect") +
  plot_annotation(title = "Pairwise Scatter Plots of Features by Activity")

# Display the plot
print(combined_plot)









# how behaviours change over time ####

# plot how the behaviours change over time
data2 <- data_other %>% arrange(ID, Timestamp) %>% slice_head(n = 940000) %>% ungroup() %>%
  mutate(numeric_activity = as.numeric(factor(Activity)), 
         Activity = factor(Activity),
         relative_seconds = row_number())

ggplot(data2, aes(x = (relative_seconds/20), y = as.numeric(numeric_activity))) +
  geom_line() +
  theme_minimal() +
  labs(
    x = "Time (seconds)",
    y = "Activity",
    color = "Activity"
  ) +
  scale_y_continuous(
    breaks = unique(data2$numeric_activity),
    labels = levels(data2$Activity)
  )

# average duration of each behaviour before it changes ####
# will create a negative number when the ID changes. Therefore I just remove all of those rows
sample_rate <- 20
summary <- data_other %>%
  arrange(ID, time) %>%
  group_by(ID) %>%
  mutate(
    # Detect behavior change
    behavior_change = lag(Activity) != Activity,
    behavior_change = ifelse(is.na(behavior_change), TRUE, behavior_change),  # Set NA for the first row of each ID
    row = row_number()
  ) %>%
  ungroup() %>%
  filter(behavior_change) %>%
  mutate(
    duration_samples = row - lag(row, default = 0),
    duration_seconds = duration_samples / sample_rate
  ) %>%
  filter(duration_seconds >= 0) # %>%
#  group_by(Activity) %>%
#  summarise(
#    average_duration = mean(duration_seconds, na.rm = TRUE),
#    variation_duration = sd(duration_seconds, na.rm = TRUE),
#    count = n()
#  )

# plot that
ggplot(summary, aes(x = Activity, y = duration_seconds)) +
  geom_boxplot(aes(color = Activity), outlier.shape = NA) +  # Use color to distinguish activities
  theme_minimal() +
  theme(legend.position = "none") +
  labs(
    x = "Activity",
    y = "Duration (seconds)"
  ) +
  scale_y_continuous(
    limits = c(min(summary$duration_seconds, na.rm = TRUE), max(summary$duration_seconds, na.rm = TRUE)),  # Set y-axis limits
    breaks = seq(0, max(summary$duration_seconds, na.rm = TRUE), by = 160)  # Adjust the step size as needed
  )



# look at the trace shapes ####
beh_trace_plot <- plot_behaviours(behaviours = unique(data_other$Activity), data = data_other, n_samples = 200, n_col = 2)
