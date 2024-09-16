# ---------------------------------------------------------------------------
# Functions for plotting
# ---------------------------------------------------------------------------


# set colours ####
generate_random_colors <- function(n) {
  colors <- rgb(runif(n), runif(n), runif(n))
  return(colors)
}
set.seed(123)

# total volume by Activity and ID ####
plot_activity_ID <- function(data, frequency, colours) {
  my_colours <- generate_random_colors(colours)
  # summarise into a table
  labelledDataSummary <- data %>%
    #filter(!Activity %in% ignore_behaviours) %>%
    count(ID, Activity)
  
  # account for the HZ, convert to minutes
  labelledDataSummaryplot <- labelledDataSummary %>%
    mutate(minutes = (n/frequency)/60)
  
  # Plot the stacked bar graph
  behaviourIndividualDistribution <- ggplot(labelledDataSummaryplot, aes(x = Activity, y = minutes, fill = as.factor(ID))) +
    geom_bar(stat = "identity") +
    labs(x = "Activity",
         y = "minutes") +
    theme_minimal() +
    scale_fill_manual(values = my_colours) +
    theme(axis.line = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.border = element_rect(color = "black", fill = NA),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())
  
  return(behaviourIndividualDistribution)
}



# DISPLAYING SAMPLES OF EACH TRACE TYPE ####
plot_trace_example <- function(behaviours, data, n_samples, n_col) {
  # Function to create the plot for each behavior
  plot_behaviour <- function(behaviour, n_samples) {
    df <- data %>%
      filter(Activity == behaviour) %>%
      group_by(ID, Activity) %>%
      slice(1:n_samples) %>%
      mutate(relative_time = row_number())
    
    
    ggplot(df, aes(x = relative_time)) +
      geom_line(aes(y = Accelerometer.X, color = "X"), show.legend = FALSE) +
      geom_line(aes(y = Accelerometer.Y, color = "Y"), show.legend = FALSE) +
      geom_line(aes(y = Accelerometer.Z, color = "Z"), show.legend = FALSE) +
      labs(title = paste(behaviour),
           x = NULL, y = NULL) +
      scale_color_manual(values = c(X = "salmon", Y = "turquoise", Z = "darkblue"), guide = "none") +
      facet_wrap(~ ID, nrow = 1, scales = "free_x") +
      theme_minimal() +
      theme(panel.grid = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_blank())
  }
  
  # Create plots for each behavior (with error catching)
  plots <- purrr::map(behaviours, function(behaviour) {
    tryCatch(
      {
        plot_behaviour(behaviour, n_samples)
      },
      error = function(e) {
        message("Skipping plot for ", behaviour, ": ", e$message)
        NULL  # Return NULL to indicate skipping
      }
    )
  })
  
  # Combine plots into a single grid
  # plots <- plots[1:13]
  grid_plot <- cowplot::plot_grid(plotlist = plots, ncol = n_col)
  
  return(grid_plot)
}


# average duration of behaviours ####
average_duration <- function(data, sample_rate){
  
  summary <- data %>%
    arrange(ID) %>%                     # Sort by ID (not time because multiple trials)
    group_by(ID) %>%                         # Group by ID
    mutate(
      behavior_change = lag(Activity) != Activity,  # Detect changes in Activity
      behavior_change = ifelse(is.na(behavior_change), TRUE, behavior_change)  # Handle the first row
    ) %>%
    mutate(
      behavior_id = cumsum(behavior_change)  # Create an identifier for each continuous behavior segment
    ) %>%
    group_by(ID, behavior_id) %>%            # Group by ID and behavior_id
    mutate(
      row_count = row_number()                # Count rows within each behavior segment
    ) %>%
    ungroup() %>%
    select(ID, Time, Activity, row_count, behavior_id) %>%   # Select relevant columns
    group_by(ID, Activity, behavior_id) %>%
    summarise(duration_sec = max(row_count)/100)
  
  duration_stats <- summary %>%
    group_by(Activity) %>%
    summarise(
      average_duration = mean(duration_sec, na.rm = TRUE),
      variation_duration = sd(duration_sec, na.rm = TRUE),
      minimum_duration = min(duration_sec, na.rm = TRUE),
      third_quartile_duration = quantile(duration_sec, 0.25, na.rm = TRUE),  # Add third quartile
      count = n()
    )

  # plot that
  duration_plot <- ggplot(summary, aes(x = Activity, y = duration_sec)) +
    geom_boxplot(aes(color = Activity)) +  # Use color to distinguish activities
    theme_minimal() +
    theme(
      legend.position = "none",             # Remove legend
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),  # Rotate x-axis labels 90 degrees
      panel.grid = element_blank(),         # Remove grid lines
      panel.border = element_rect(color = "black", fill = NA)  # Add black border around the plot
    ) +
    labs(
      x = "Activity",
      y = "Duration (seconds)"
    ) +
    scale_y_continuous(
      limits = c(min(summary$duration_sec, na.rm = TRUE), max(summary$duration_sec, na.rm = TRUE)),  # Set y-axis limits
      breaks = seq(0, max(summary$duration_sec, na.rm = TRUE), by = 160)  # Adjust the step size as needed
    )

  return(list(duration_plot = duration_plot,
              duration_stats = duration_stats))
}

