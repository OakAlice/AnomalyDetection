# load in the data and select the relevant columns ####
dog_data <- fread(file.path(base_path, "Data", "Vehkaoja_Dog.csv"))
dog_data2 <- dog_data %>%
  select("DogID", "t_sec", "ANeck_x", "ANeck_y", "ANeck_z", "Behavior_1", "Behavior_2", "Behavior_3") %>%
  rename(
    "ID" = "DogID",
    "Time" = "t_sec",
    "Accelerometer.X" = "ANeck_x",
    "Accelerometer.Y" = "ANeck_y",
    "Accelerometer.Z" = "ANeck_z",
  )

# remove the non-behaviours and create compound labels ####
dog_data3 <- dog_data2 %>%
  mutate(across(where(is.character), ~ na_if(., "<undefined>"))) %>%
  mutate(across(where(is.character), ~ na_if(., "Extra_Synchronization"))) %>%
  mutate(across(where(is.character), ~ na_if(., "Synchronization"))) %>%
  mutate(
    Label_1_2 = paste(Behavior_1, Behavior_2, sep = "_"),
    Label_2_3 = paste(Behavior_2, Behavior_3, sep = "_"),
    Label_1_3 = paste(Behavior_1, Behavior_3, sep = "_"),
    Label_1_2_3 = paste(Behavior_1, Behavior_2, Behavior_3, sep = "_")
  )

summary <- dog_data3 %>%
  group_by(
    Behavior_1, Behavior_2, Behavior_3, Label_1_2,
    Label_2_3, Label_1_3, Label_1_2_3
  ) %>%
  count()

# extract data for each of the Beh 1 and plot ####
Walking_data <- dog_data3 %>%
  filter(ID %in% unique(dog_data3$ID)[1:5]) %>%
  filter(Behavior_1 == "Sniffing")

# plot
plotTraceExamples(
  behaviours = unique(Walking_data$Label_1_2_3),
  data = Walking_data, n_samples = 200, n_col = 2,
  label_choice = "Label_1_2_3"
)
# function
plotTraceExamples <- function(behaviours, data, n_samples, n_col, label_choice) {
  # Function to create the plot for each behavior
  plot_behaviour <- function(behaviour, n_samples, label_choice) {
    df <- data %>%
      filter(!!sym(label_choice) == behaviour) %>%
      group_by(ID, !!sym(label_choice)) %>%
      slice(1:n_samples) %>%
      mutate(relative_time = row_number())

    ggplot(df, aes(x = relative_time)) +
      geom_line(aes(y = Accelerometer.X, color = "X"), show.legend = FALSE) +
      geom_line(aes(y = Accelerometer.Y, color = "Y"), show.legend = FALSE) +
      geom_line(aes(y = Accelerometer.Z, color = "Z"), show.legend = FALSE) +
      labs(
        title = paste(behaviour),
        x = NULL, y = NULL
      ) +
      scale_color_manual(values = c(X = "salmon", Y = "turquoise", Z = "darkblue"), guide = "none") +
      facet_wrap(~ID, nrow = 1, scales = "free_x") +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_blank()
      )
  }

  # Create plots for each behavior (with error catching)
  plots <- purrr::map(behaviours, function(behaviour) {
    tryCatch(
      {
        plot_behaviour(behaviour, n_samples, label_choice)
      },
      error = function(e) {
        message("Skipping plot for ", behaviour, ": ", e$message)
        NULL # Return NULL to indicate skipping
      }
    )
  })

  # Remove NULL plots
  plots <- Filter(Negate(is.null), plots)

  # Combine plots into a single grid
  grid_plot <- cowplot::plot_grid(plotlist = plots, ncol = n_col)

  return(grid_plot)
}

# dominant and recessive rules ####
# based on inspection of these plots, behaviours were considered
# dominant (most responsible for trace shape), or recessive (less responsible)
# in a perfect world, each sub-category would be a category, but not here
# therefore, overwrite recessive behaviours

dog_data4 <- dog_data3 %>%
  mutate(
    Activity =
      ifelse(Behavior_1 %in% c("Walking", "Eating", "Trotting", "Tugging", "Galloping", "Panting"),
        Behavior_1, ifelse(is.na(Behavior_2), Behavior_1, Behavior_2)
      )
  ) %>%
  select(ID, Activity, Time, Accelerometer.X, Accelerometer.Y, Accelerometer.Z) %>%
  na.omit()

# write this out
fwrite(dog_data4, file.path(base_path, "Data", "Vehkaoja_Dog.csv"))


summary <- dog_data4 %>%
  group_by(Activity) %>%
  count()

# stitch all the feature data together
files <- list.files(file.path(base_path, "Data", "Feature_data", "Vehkaoja_Dog"), full.names = TRUE)

combined_data <- files %>%
  lapply(fread) %>%
  bind_rows()

fwrite(combined_data, file.path(base_path, "Data", "Feature_data", "Vehkaoja_Dog_other_features.csv"))
