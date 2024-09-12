# ---------------------------------------------------------------------------
# Preprocessing Decisions
# Script for exploring data to determine behavioural groupings and window length
# ---------------------------------------------------------------------------


# generate plots
# volume of data per individual and activity
plot_activity_ID_graph <- plot_activity_ID(data = move_data, 
                                           frequency = movement_data$Frequency, colours = length(unique(move_data$ID)))
# examples of each trace
plot_trace_example_graph <- plot_trace_example(behaviours = unique(move_data$Activity), 
                                               data = move_data, n_samples = 200, n_col = 4)

# other exploratory plots
## Prt 1: Determine Window Length ####
duration_info <- average_duration(data = data_other, sample_rate = movement_data$Frequency)
stats <- duration_info$duration_stats %>% 
  filter(Activity %in% movement_data$target_behaviours)
plot <- duration_info$duration_plot
