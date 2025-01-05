# Function to calculate binary metrics
calculate_binary_metrics <- function(true_labels, pred_labels) {
  # Convert predictions to TRUE/FALSE
  true_labels <- true_labels != "Other"
  pred_labels <- pred_labels != "Other"
  
  # Calculate metrics
  precision <- MLmetrics::Precision(y_pred = pred_labels, y_true = true_labels, positive = TRUE)
  recall <- MLmetrics::Recall(y_pred = pred_labels, y_true = true_labels, positive = TRUE)
  f1 <- MLmetrics::F1_Score(y_pred = pred_labels, y_true = true_labels, positive = TRUE)
  accuracy <- MLmetrics::Accuracy(y_pred = pred_labels, y_true = true_labels)
  
  return(c(Precision = precision, Recall = recall, F1_Score = f1, Accuracy = accuracy))
}


# Load in the data --------------------------------------------------------
ground_truth <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv"))) %>%
  select(Time, ID, Activity, OtherActivity, GeneralisedActivity)

occ_files <- list.files(file.path(base_path, "Output", "Testing", "Predictions"), 
                        pattern = paste0(dataset_name, ".*_OCC_.*\\.csv$"), 
                        full.names = TRUE)

occ_data <- lapply(occ_files, function(file) {
  df <- fread(file)
  df[, model_activity := str_extract(basename(file), "(?<=OCC_)[^_]+(?=_predictions)")]
  return(df)
}) %>% rbindlist()

# Load and process Binary predictions ----------------------------------------
binary_files <- list.files(file.path(base_path, "Output", "Testing", "Predictions"), 
                           pattern = paste0(dataset_name, ".*_Binary_.*\\.csv$"), 
                           full.names = TRUE)

binary_data <- lapply(binary_files, function(file) {
  df <- fread(file)
  df[, model_activity := str_extract(basename(file), "(?<=Binary_)[^_]+(?=_predictions)")]
  return(df)
}) %>% rbindlist()


# Process OCC predictions -------------------------------------------------
setDT(occ_data)

# Pivot the data to widen it
occ_wide <- dcast(
  occ_data, 
  Time + ID + Ground_truth ~ model_activity, 
  value.var = "Predictions", 
  fun.aggregate = function(x) if (length(x) > 0) x[1] else NA
)

# Rename columns to specify predictions
setnames(occ_wide, old = names(occ_wide)[-(1:3)], 
         new = paste0(names(occ_wide)[-(1:3)], "_prediction"))

# make the parent prediction
occ_wide <- occ_wide %>%
  mutate(
    collective_prediction = coalesce(
      !!!lapply(select(., ends_with("_prediction")), function(x) ifelse(x != "Other", x, NA_character_)),
      "Other"
    )
  )

# add in the ground truth dataframe
ground_truth_OCC <- ground_truth %>% select(Time, ID, Activity) %>%
  mutate(Activity = ifelse(!ground_truth_OCC$Activity %in% target_activities, "Other", ground_truth_OCC$Activity))

occ_wide <- merge(occ_wide, ground_truth_OCC, by = c("Time", "ID"))


table(occ_wide$collective_prediction, occ_wide$Activity)
MLmetrics::F1_Score(occ_wide$Activity, occ_wide$collective_prediction)










# Create visualization of combined predictions ---------------------------
combined_predictions_plot <- ggplot() +
  geom_tile(data = occ_combined, 
            aes(x = Time, y = "OCC Combined", fill = Combined_pred)) +
  labs(title = "Combined Model Predictions", 
       x = "Time", 
       y = "Model Type") +
  theme_minimal() +
  theme(panel.grid = element_blank())
