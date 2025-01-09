# Ensemble performance and predictions ------------------------------------

# Ground truth data labels ------------------------------------------------
ground_truth <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv"))) %>%
  select(Time, ID, Activity, OtherActivity, GeneralisedActivity)
ground_truth_labels <- ground_truth %>% 
  select(Time, ID, Activity) %>%
  mutate(Activity = ifelse(!Activity %in% target_activities, "Other", Activity))

# Load in the data --------------------------------------------------------
# Helper function to load prediction files
load_predictions <- function(pattern) {
  files <- list.files(file.path(base_path, "Output", "Testing", "Predictions"), 
                      pattern = pattern,
                      full.names = TRUE)
  
  rbindlist(lapply(files, function(file) {
    df <- fread(file)
    df[, model_activity := gsub(".*_(Binary|OCC)_(.*?)_predictions.*", "\\2", basename(file))]
    return(df)
  }))
}

# Process predictions -----------------------------------------------------
for (model in c("OCC", "Binary")){
  test_results <- data.frame()
  print(model)
  
  data <- load_predictions(paste0(dataset_name, ".*_", model, "_.*\\.csv$"))
  
  data_wide1 <- merge(
    dcast(data, Time + ID + Ground_truth ~ model_activity, 
          value.var = "Predictions"),
    dcast(data, Time + ID + Ground_truth ~ model_activity, 
          value.var = "Decision_values"),
    by = c("Time", "ID", "Ground_truth"),
    suffixes = c("_Prediction", "_Decision")
  ) 
  
  
  data_wide <- data_wide1#[1:50,]
  
  
  data_wide_summarised <- data_wide %>% 
    rowwise() %>%
    mutate(
      collective_prediction = {
        preds <- c_across(ends_with("_Prediction"))
        # Filter out both "Other" and NA values
        valid_idx <- which(preds != "Other" & !is.na(preds))
        
        if(length(valid_idx) > 0) {
          # Get the valid predictions
          valid_preds <- preds[valid_idx]
          
          # Define priority order
          priority_order <- c("Shaking", "Walking", "Eating", "Lying chest")
          
          # Find the first matching priority behavior
          matching_priorities <- priority_order[priority_order %in% valid_preds]
          
          if(length(matching_priorities) > 0) {
            matching_priorities[1]  # Take the highest priority match
          } else {
            valid_preds[1]  # Fallback to first valid prediction
          }
          
        } else {
          "Other"
        }
      }
    ) %>%
    ungroup()
  
  # merge with the ground truth
  data_wide2 <- merge(data_wide_summarised, ground_truth_labels, by = c("Time", "ID"))
  
  # calculate performance
  collective_predictions <- data_wide2$collective_prediction
  collective_ground_truth <- data_wide2$Activity
  
  if (length(collective_predictions) != length(collective_ground_truth)) {
    stop("Error: Predictions and ground truth labels have different lengths.")
  }
  
  # Calculate per-class metrics and macro average
  test_results <- calculate_full_multi_performance(ground_truth_labels = collective_ground_truth, predictions = collective_predictions)
    
  # save the ensemble results
  fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", model, "_ensemble_test_performance.csv")))
  fwrite(data_wide, file.path(base_path, "Output", "Testing", "Predictions", paste0(dataset_name, "_", model, "_ensemble_predictions.csv")))
}









# Plotting performance ----------------------------------------------------

ground_truth <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv"))) %>%
  select(Time, ID, Activity, OtherActivity, GeneralisedActivity)
ground_truth_labels <- ground_truth %>% 
  select(Time, ID, Activity) %>%
  mutate(Activity = ifelse(!Activity %in% target_activities, "Other", Activity))


OCC_ensemble <- fread(file.path(base_path, "Output", "Testing", "Predictions", paste0(dataset_name, "_Binary_ensemble_predictions.csv"))) %>%
  select(-Ground_truth)

First_ind <- OCC_ensemble %>% filter(ID == unique(OCC_ensemble$ID)[1])


# Create the activity plot
activity_plot <- ggplot(First_ind, aes(x = Time, y = 1, fill = Activity)) +
  geom_tile() +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal() +
  theme(axis.text.y = element_blank(),
        axis.title.y = element_blank()) +
  labs(title = "Activity Over Time")

# Reshape prediction data for plotting
predictions_data <- First_ind %>%
  select(Time, ends_with("_Prediction")) %>%
  pivot_longer(
    cols = ends_with("_Prediction"),
    names_to = "Prediction_Type",
    values_to = "Prediction"
  ) %>%
  # Remove "_Prediction" from names for cleaner labels
  mutate(Prediction_Type = gsub("_Prediction", "", Prediction_Type))

Lying <- predictions_data %>% filter(Prediction_Type == "Lying chest")



# Create the predictions plot
predictions_plot <- ggplot(predictions_data, aes(x = Time, y = Prediction_Type, fill = Prediction)) +
  geom_tile() +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal() +
  labs(y = "Prediction Type",
       title = "Predictions Over Time")

# Combine plots vertically
combined_plot <- activity_plot / predictions_plot +
  plot_layout(heights = c(1, 4))

# Display the combined plot
print(combined_plot)






