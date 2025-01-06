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
test_results <- data.frame()
for (model in c("OCC", "Binary")){
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
  class_metrics <- lapply(unique_classes, function(class) {
    # Convert to binary problem for this class
    binary_true <- factor(collective_ground_truth == class, levels = c(FALSE, TRUE))
    binary_pred <- factor(collective_predictions == class, levels = c(FALSE, TRUE))
    
    # Calculate metrics for this class
    f1 <- MLmetrics::F1_Score(y_true = binary_true, y_pred = binary_pred, positive = TRUE)
    precision <- MLmetrics::Precision(y_true = binary_true, y_pred = binary_pred, positive = TRUE)
    recall <- MLmetrics::Recall(y_true = binary_true, y_pred = binary_pred, positive = TRUE)
    accuracy <- MLmetrics::Accuracy(y_true = binary_true, y_pred = binary_pred)
    
    # Return as named vector
    c(
      Class = as.character(class),
      F1_Score = f1,
      Precision = precision,
      Recall = recall,
      Accuracy = accuracy
    )
  })
  
  # Combine per-class metrics into a data frame - first turn each class into a row
  class_metrics_df <- do.call(rbind, lapply(class_metrics, function(x) {
    # Convert the named vector to a one-row data frame
    as.data.frame(t(x), stringsAsFactors = FALSE)
  }))
  # convert all to numeric
  class_metrics_df$F1_Score <- as.numeric(class_metrics_df$F1_Score)
  class_metrics_df$Precision <- as.numeric(class_metrics_df$Precision)
  class_metrics_df$Recall <- as.numeric(class_metrics_df$Recall)
  class_metrics_df$Accuracy <- as.numeric(class_metrics_df$Accuracy)
  
  # Calculate macro averages (replacing NA with 0)
  macro_metrics <- colMeans(replace(class_metrics_df[, c("F1_Score", "Precision", "Recall", "Accuracy")], 
                                    is.na(class_metrics_df[, c("F1_Score", "Precision", "Recall", "Accuracy")]), 
                                    0))
  
  # Compile results for macro averages
  macro_results <- data.frame(
    Dataset = as.character(dataset_name),
    Model = model,
    Activity = "MacroAverage",
    F1_Score = as.numeric(macro_metrics["F1_Score"]),
    Precision = as.numeric(macro_metrics["Precision"]),
    Recall = as.numeric(macro_metrics["Recall"]),
    Accuracy = as.numeric(macro_metrics["Accuracy"])
  )
  
  # Add per-class results
  class_level_results <- data.frame(
    Dataset = rep(as.character(dataset_name), length(unique_classes)),
    Model = rep(model, length(unique_classes)),
    Activity = class_metrics_df$Class,
    F1_Score = class_metrics_df$F1_Score,
    Precision = class_metrics_df$Precision,
    Recall = class_metrics_df$Recall,
    Accuracy = class_metrics_df$Accuracy
  )
  
  # Combine macro-average and per-class results
  test_results <- rbind(test_results, macro_results, class_level_results)
  
  # save the ensemble results
  fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", model, "_ensemble_test_performance.csv")))
  fwrite(data_wide, file.path(base_path, "Output", "Testing", "Predictions", paste0(dataset_name, "_", model, "_ensemble_predictions.csv")))
}
