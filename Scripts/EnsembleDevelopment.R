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
  
  data_wide <- merge(
    dcast(data, Time + ID + Ground_truth ~ model_activity, 
          value.var = "Predictions"),
    dcast(data, Time + ID + Ground_truth ~ model_activity, 
          value.var = "Decision_values"),
    by = c("Time", "ID", "Ground_truth"),
    suffixes = c("_Prediction", "_Decision")
  ) 
  
  data_wide_summarised <- data_wide %>% 
    rowwise() %>%
    mutate(
      collective_prediction = {
        preds <- c_across(ends_with("_Prediction"))
        scores <- c_across(ends_with("_Decision"))
        valid_idx <- which(preds != "Other")
        
        if(length(valid_idx) > 0) {
          preds[valid_idx[which.max(abs(scores[valid_idx]))]]
        } else {
          # Default to "Other" if no valid predictions
          "Other"
        }
      }
    ) %>%
    ungroup()
  
  # merge with the ground truth
  data_wide <- merge(data_wide_summarised, ground_truth_labels, by = c("Time", "ID"))
  
  # calculate performance
  collective_predictions <- data_wide$collective_prediction
  collective_ground_truth <- data_wide$Activity
  
  unique_classes <- sort(union(collective_predictions, collective_ground_truth))
  collective_predictions <- factor(collective_predictions, levels = unique_classes)
  collective_ground_truth <- factor(collective_ground_truth, levels = unique_classes)
  
  if (length(collective_predictions) != length(collective_ground_truth)) {
    stop("Error: Predictions and ground truth labels have different lengths.")
  }
  
  # Calculate per-class metrics and macro average
  ensemble_results <- multiclass_class_metrics(collective_ground_truth, collective_predictions)
  class_metrics_df <- ensemble_results$class_metrics
  macro_metrics <- ensemble_results$macro_metrics
  
  class_metrics_df <- do.call(rbind, lapply(class_metrics_df, function(x) {
    # Convert any "NaN" to NA
    x[x == "NaN"] <- NA
    # Convert numeric strings to actual numbers
    x[-1] <- as.numeric(x[-1])  # Keep first column (Class) as character
    as.data.frame(t(x), stringsAsFactors = FALSE)
  }))
  
  # Get baseline metrics
  # 1. Zero Rate (always predict the majority class)
  majority_class <- names(which.max(table(collective_ground_truth)))
  zero_rate_preds <- factor(
    rep(majority_class, length(collective_ground_truth)),
    levels = levels(as.factor(collective_ground_truth))
  )
  zero_rate_baseline <- multiclass_class_metrics(collective_ground_truth, zero_rate_preds)
  #### extract the per-class values ####
  macro_metrics_zero <- zero_rate_baseline$macro_metrics
  class_metrics_zero <- zero_rate_baseline$class_metrics
  class_metrics_zero <- do.call(rbind, lapply(class_metrics_zero, function(x) {
    as.data.frame(t(x), stringsAsFactors = FALSE)
  }))
  
  # 2. Random baseline (randomly select in stratified proportion to true data)
  random_multiclass <- random_baseline_metrics(collective_ground_truth, iterations = 100)
  random_macro_summary <- random_multiclass$macro_summary
  random_class_summary <- random_multiclass$class_summary$averages
  random_class_summary <- as.data.frame(random_class_summary)
  
  # Compile results for macro averages
  macro_results <- data.frame(
    Dataset = as.character(dataset_name),
    Model = paste0(model, "_Ensemble"),
    Activity = "MacroAverage",
    F1_Score = as.numeric(macro_metrics["F1_Score"]),
    Precision = as.numeric(macro_metrics["Precision"]),
    Recall = as.numeric(macro_metrics["Recall"]),
    Accuracy = as.numeric(macro_metrics["Accuracy"]),
    Random_F1_Score = random_macro_summary$averages["F1_Score"],
    Random_Precision = random_macro_summary$averages["Precision"],
    Random_Recall = random_macro_summary$averages["Recall"],
    Random_Accuracy = random_macro_summary$averages["Accuracy"],
    ZeroR_F1_Score = macro_metrics_zero$F1_Score,
    ZeroR_Precision = macro_metrics_zero$Precision,
    ZeroR_Recall = macro_metrics_zero$Recall,
    ZeroR_Accuracy = macro_metrics_zero$Accuracy
  )
  
  activity_results_list <- list()
  # Loop through each unique activity
  for (activity in unique(collective_ground_truth)) {
    # Create dataframe for current activity
    activity_results_list[[activity]] <- data.frame(
      Dataset = dataset_name,
      Model = paste0(model, "_Ensemble"),
      Activity = activity,
      F1_Score = class_metrics_df$F1_Score[class_metrics_df$Class == activity],
      Precision = class_metrics_df$Precision[class_metrics_df$Class == activity],
      Recall = class_metrics_df$Recall[class_metrics_df$Class == activity],
      Accuracy = class_metrics_df$Accuracy[class_metrics_df$Class == activity],
      Random_F1_Score = random_class_summary[activity, "F1_Score"],
      Random_Precision = random_class_summary[activity, "Precision"],
      Random_Recall = random_class_summary[activity, "Recall"],
      Random_Accuracy = random_class_summary[activity, "Accuracy"],
      ZeroR_F1_Score = class_metrics_zero$F1_Score[class_metrics_zero$Class == activity],
      ZeroR_Precision = class_metrics_zero$Precision[class_metrics_zero$Class == activity],
      ZeroR_Recall = class_metrics_zero$Recall[class_metrics_zero$Class == activity],
      ZeroR_Accuracy = class_metrics_zero$Accuracy[class_metrics_zero$Class == activity]
    )
  }
  
  final_class_results <- do.call(rbind, activity_results_list)
  
  # Combine macro-average and per-class results
  test_results <- rbind(test_results, macro_results, final_class_results)
  
  # save the ensemble results
  fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", model, "_ensemble_test_performance.csv")))
  fwrite(data_wide, file.path(base_path, "Output", "Testing", "Predictions", paste0(dataset_name, "_", model, "_ensemble_predictions.csv")))
}
