# Zero rate Classification for dichotomous models -------------------------
calculate_zero_rate_baseline <- function(ground_truth_labels, model_type, target_class = NULL) {
  if (model_type == "OCC" | model_type == "Binary") {
    # For OCC and Binary, always predict the target class
    zero_rate_preds <- factor(
      rep(target_class, length(ground_truth_labels)),
      levels = levels(ground_truth_labels)
    )
    # Calculate metrics
    f1 <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = zero_rate_preds, positive = target_class)
    precision <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = zero_rate_preds, positive = target_class)
    recall <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = zero_rate_preds, positive = target_class)
    accuracy <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = zero_rate_preds)
    
  } else {
    print("this doesn't work for multiclass")
  }
  # Return metrics
  c(
    F1_Score = f1,
    Precision = precision,
    Recall = recall,
    Accuracy = accuracy
  )
}


# Unadjusted performance of model -----------------------------------------
multiclass_class_metrics <- function(ground_truth_labels, predictions) { 
  # Ensure predictions and ground_truth_labels are factors with the same levels
  unique_classes <- sort(union(predictions, ground_truth_labels))
  predictions <- factor(predictions, levels = unique_classes)
  ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)
  
  # Calculate class prevalence
  prevalence <- table(ground_truth_labels) / length(ground_truth_labels)
  prevalence_df <- data.frame(Class = names(prevalence), Prevalence = as.numeric(prevalence))
  
  # Calculate per-class metrics
  class_metrics <- lapply(unique_classes, function(class) {
    binary_true <- factor(ground_truth_labels == class, levels = c(FALSE, TRUE))
    binary_pred <- factor(predictions == class, levels = c(FALSE, TRUE))
    
    c(
      Class = as.character(class),
      F1_Score = MLmetrics::F1_Score(y_true = binary_true, y_pred = binary_pred, positive = TRUE),
      Precision = MLmetrics::Precision(y_true = binary_true, y_pred = binary_pred, positive = TRUE),
      Recall = MLmetrics::Recall(y_true = binary_true, y_pred = binary_pred, positive = TRUE),
      Accuracy = MLmetrics::Accuracy(y_true = binary_true, y_pred = binary_pred)
    )
  })
  
  # Convert metrics to data frame
  class_metrics_df <- do.call(rbind, lapply(class_metrics, function(x) {
    as.data.frame(t(x), stringsAsFactors = FALSE)
  })) %>%
    mutate(across(c(F1_Score, Precision, Recall, Accuracy), as.numeric))
  
  # Merge with prevalence data
  class_metrics_df <- class_metrics_df %>%
    left_join(prevalence_df, by = "Class")
  
  is.nan.data.frame <- function(x) { # sourced from stack excahnge
    do.call(cbind, lapply(x, is.nan))
  }
  class_metrics_df[is.nan.data.frame(class_metrics_df)] <- 0
  
  # Calculate weighted averages
  weighted_metrics <- class_metrics_df %>%
    summarize(across(c(F1_Score, Precision, Recall, Accuracy), ~ sum(.x * prevalence_df$Prevalence, na.rm = TRUE))) %>%
    as.list()
  
  # Calculate macro averages
  macro_metrics <- colMeans(replace(class_metrics_df[, c("F1_Score", "Precision", "Recall", "Accuracy")], 
                                    is.na(class_metrics_df[, c("F1_Score", "Precision", "Recall", "Accuracy")]), 
                                    0)) %>% as.list()
  
  return(list(
    macro_metrics = macro_metrics,
    weighted_metrics = weighted_metrics,
    class_metrics = class_metrics_df
  ))
}


# Format the test data for testing model ----------------------------------
prepare_test_data <- function(test_feature_data, selected_features, behaviour = NULL) {
  # Select features and metadata columns
  selected_data <- test_feature_data[, .SD, .SDcols = c(selected_features, "Activity", "Time", "ID")]
  selected_data <- na.omit(as.data.table(selected_data))
  
  # For dichotomous models, convert to binary classification
  if (!is.null(behaviour)) {
    selected_data$Activity <- ifelse(selected_data$Activity == behaviour, behaviour, "Other")
  }
  
  # Extract labels and metadata
  ground_truth_labels <- selected_data$Activity
  time_values <- selected_data$Time
  ID_values <- selected_data$ID
  
  # Get numeric features only
  numeric_data <- selected_data[, !c("Activity", "Time", "ID"), with = FALSE]
  
  # Remove invalid rows
  invalid_rows <- which(!complete.cases(numeric_data) |
                          !apply(numeric_data, 1, function(row) all(is.finite(row))))
  
  if (length(invalid_rows) > 0) {
    numeric_data <- numeric_data[-invalid_rows, , drop = FALSE]
    ground_truth_labels <- ground_truth_labels[-invalid_rows]
    time_values <- time_values[-invalid_rows]
    ID_values <- ID_values[-invalid_rows]
  }
  
  list(
    numeric_data = numeric_data,
    ground_truth_labels = ground_truth_labels,
    time_values = time_values,
    ID_values = ID_values
  )
}

save_results <- function(results, predictions, ground_truth_labels, time_values, ID_values,
                         dataset_name, model_name, base_path) {
  # Save performance metrics
  fwrite(results, file.path(base_path, "Output", "Testing", 
                            paste0(dataset_name, "_", model_name, "_test_performance.csv")))
  
  # Save predictions
  output <- data.table(
    "Time" = time_values,
    "ID" = ID_values,
    "Ground_truth" = ground_truth_labels,
    "Predictions" = predictions
  )
  
  if (nrow(output) > 0) {
    fwrite(output, file.path(base_path, "Output", "Testing", "Predictions",
                             paste(dataset_name, model_name, "predictions.csv", sep = "_")))
  }
}



# Random baseline for multiclass metrics ----------------------------------
random_baseline_metrics <- function(ground_truth_labels, iterations = 100) {
  # Convert ground truth to factor and get class information
  ground_truth_labels <- factor(ground_truth_labels)
  class_levels <- levels(ground_truth_labels)
  class_props <- prop.table(table(ground_truth_labels))
  n_classes <- length(class_levels)
  n_samples <- length(ground_truth_labels)
  
  # Pre-allocate matrices for storing results
  class_metrics <- list()
  # Run iterations
  for (i in 1:iterations) {
    # Generate random predictions
    random_preds <- factor(
      sample(class_levels, size = n_samples, prob = class_props, replace = TRUE),
      levels = class_levels
    )
    
    random_baseline <- multiclass_class_metrics(ground_truth_labels, random_preds)
    random_baseline_class <- random_baseline$class_metrics
    
    # Store class metrics
    class_metrics[[i]] <- random_baseline_class
    }
    
  # Calculate average for each class and metric
  class_metrics_combined <- do.call(rbind, class_metrics)
  
  averages_df <- class_metrics_combined %>%
    group_by(Class) %>%
    summarise(
      F1_Score = mean(F1_Score, na.rm = TRUE),
      Precision = mean(Precision, na.rm = TRUE),
      Recall = mean(Recall, na.rm = TRUE),
      Accuracy = mean(Accuracy, na.rm = TRUE),
      Prevalence = mean(Prevalence, na.rm = TRUE)
    )
  
  # calculate the weighted average from the individual classes
  weighted_metrics <- averages_df %>%
    summarize(across(c(F1_Score, Precision, Recall, Accuracy), ~ sum(.x * averages_df$Prevalence, na.rm = TRUE))) %>%
    as.list()
  
  # Calculate summary statistics
  list(
    macro_summary = list(
      F1_Score = weighted_metrics$F1_Score,
      Precision = weighted_metrics$Precision,
      Recall = weighted_metrics$Recall,
      Accuracy = weighted_metrics$Accuracy
    ),
    class_summary = averages_df
  )
}



# Unadjusted performance, random, and zero baseline -----------------------
calculate_full_multi_performance <- function(ground_truth_labels, predictions){
  
  # 1. absolute scores (unadjusted)
  macro_multiclass_scores <- multiclass_class_metrics(ground_truth_labels, predictions)
  class_metrics_df <- macro_multiclass_scores$class_metrics
  macro_metrics <- macro_multiclass_scores$macro_metrics
  weighted_metrics <- macro_multiclass_scores$weighted_metrics
  
  # 2. Zero Rate baseline (always predict the majority class)
  majority_class <- names(which.max(table(ground_truth_labels)))
  zero_rate_preds <- factor(
    rep(majority_class, length(ground_truth_labels)),
    levels = levels(as.factor(ground_truth_labels))
  )
  zero_rate_baseline <- multiclass_class_metrics(ground_truth_labels, zero_rate_preds)
  macro_metrics_zero <- zero_rate_baseline$macro_metrics
  class_metrics_zero <- zero_rate_baseline$class_metrics
  
  # 3. Random baseline (randomly select in stratified proportion to true data)
  random_multiclass <- random_baseline_metrics(ground_truth_labels, iterations = 100)
  random_macro_summary <- random_multiclass$macro_summary
  random_class_summary <- random_multiclass$class_summary
  
  # Compile results
  macro_results <- data.frame(
    Dataset = dataset_name,
    Model = behaviour_set,
    Activity = "WeightedMacroAverage",
    F1_Score = weighted_metrics["F1_Score"],
    Precision = weighted_metrics["Precision"],
    Recall = weighted_metrics["Recall"],
    Accuracy = weighted_metrics["Accuracy"],
    Random_F1_Score = random_macro_summary$F1_Score,
    Random_Precision = random_macro_summary$Precision,
    Random_Recall = random_macro_summary$Recall,
    Random_Accuracy = random_macro_summary$Accuracy,
    ZeroR_F1_Score = macro_metrics_zero$F1_Score,
    ZeroR_Precision = macro_metrics_zero$Precision,
    ZeroR_Recall = macro_metrics_zero$Recall,
    ZeroR_Accuracy = macro_metrics_zero$Accuracy
  )
  
  activity_results_list <- list()
  # Loop through each unique activity
  for (activity in unique(ground_truth_labels)) {
    # Create dataframe for current activity
    activity_results_list[[activity]] <- data.frame(
      Dataset = dataset_name,
      Model = behaviour_set,
      Activity = activity,
      
      F1_Score = class_metrics_df$F1_Score[class_metrics_df$Class == activity],
      Precision = class_metrics_df$Precision[class_metrics_df$Class == activity],
      Recall = class_metrics_df$Recall[class_metrics_df$Class == activity],
      Accuracy = class_metrics_df$Accuracy[class_metrics_df$Class == activity],
      
      Random_F1_Score = random_class_summary$F1_Score[random_class_summary$Class == activity],
      Random_Precision = random_class_summary$Precision[random_class_summary$Class == activity],
      Random_Recall = random_class_summary$Recall[random_class_summary$Class == activity],
      Random_Accuracy = random_class_summary$Accuracy[random_class_summary$Class == activity],
      
      ZeroR_F1_Score = class_metrics_zero$F1_Score[class_metrics_zero$Class == activity],
      ZeroR_Precision = class_metrics_zero$Precision[class_metrics_zero$Class == activity],
      ZeroR_Recall = class_metrics_zero$Recall[class_metrics_zero$Class == activity],
      ZeroR_Accuracy = class_metrics_zero$Accuracy[class_metrics_zero$Class == activity]
    )
  }
  
  # Combine all results into a single dataframe
  final_class_results <- do.call(rbind, activity_results_list)
  
  # Combine all results
  test_results <- rbind(test_results, macro_results, final_class_results)
}
