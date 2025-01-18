# Zero rate Classification for dichotomous models -------------------------
calculate_zero_rate_baseline <- function(ground_truth_labels, model_type, target_class = NULL) {
  if (model_type == "OCC" | model_type == "Binary") {
    # For OCC and Binary, always predict the non-target class
    zero_rate_preds <- factor(
      rep("Other", length(ground_truth_labels)),
      levels = levels(ground_truth_labels)
    )
    # Calculate metrics
    f1 <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = zero_rate_preds, positive = target_class)
    precision <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = zero_rate_preds, positive = target_class)
    recall <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = zero_rate_preds, positive = target_class)
    accuracy <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = zero_rate_preds)
    
  } else {
    print("this doesn't work for multiclass, use other one")
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
  
  # Calculate macro averages (not weighted)
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
  # if (!is.null(behaviour)) {
  #  selected_data$Activity <- ifelse(selected_data$Activity == behaviour, behaviour, "Other")
  # }
  
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
random_baseline_metrics <- function(ground_truth_labels, iterations = 100, model = "Multi") {
  # Convert ground truth to factor and get class information
  ground_truth_labels <- factor(ground_truth_labels)
  class_levels <- levels(ground_truth_labels)
  class_props <- prop.table(table(ground_truth_labels))
  n_classes <- length(class_levels)
  n_samples <- length(ground_truth_labels)
  
  class_prop_equal <- rep(1/n_classes, n_classes)
  
  # Pre-allocate matrices for storing results
  class_metrics_prev <- list()
  class_metrics_equal <- list()
  # Run iterations
  for (i in 1:iterations) {
    # Generate random predictions
    random_preds_prev <- factor(
      sample(class_levels, size = n_samples, prob = class_props, replace = TRUE),
      levels = class_levels
    )
    
    random_preds_equal <- factor(
      sample(class_levels, size = n_samples, prob = class_prop_equal, replace = TRUE),
      levels = class_levels
    )
    
    random_baseline_prev <- multiclass_class_metrics(ground_truth_labels, random_preds_prev)
    random_baseline_class_prev <- random_baseline_prev$class_metrics
    
    random_baseline_equal <- multiclass_class_metrics(ground_truth_labels, random_preds_equal)
    random_baseline_class_equal <- random_baseline_equal$class_metrics
    
    # Store class metrics
    class_metrics_prev[[i]] <- random_baseline_class_prev
    class_metrics_equal[[i]] <- random_baseline_class_equal
    }
    
  # Calculate average for each class and metric
  class_metrics_combined_prev <- do.call(rbind, class_metrics_prev)
  class_metrics_combined_equal <- do.call(rbind, class_metrics_equal)

  averages_df_prev <- class_metrics_combined_prev %>%
    group_by(Class) %>%
    summarise(
      F1_Score = mean(F1_Score, na.rm = TRUE),
      Precision = mean(Precision, na.rm = TRUE),
      Recall = mean(Recall, na.rm = TRUE),
      Accuracy = mean(Accuracy, na.rm = TRUE),
      Prevalence = mean(Prevalence, na.rm = TRUE)
    )
  
  averages_df_equal <- class_metrics_combined_equal %>%
    group_by(Class) %>%
    summarise(
      F1_Score = mean(F1_Score, na.rm = TRUE),
      Precision = mean(Precision, na.rm = TRUE),
      Recall = mean(Recall, na.rm = TRUE),
      Accuracy = mean(Accuracy, na.rm = TRUE),
      Prevalence = mean(Prevalence, na.rm = TRUE)
    )
  
  if(model == "OCC" | model == "Binary"){
    selected_metrics_prev <- averages_df_prev[!averages_df_prev$Class == "Other", ]
    selected_metrics_equal <- averages_df_equal[!averages_df_equal$Class == "Other", ]
    
  } else {
    # calculate the weighted average from the individual classes
  selected_metrics_prev <- averages_df_prev %>%
    summarize(across(c(F1_Score, Precision, Recall, Accuracy), ~ sum(.x * averages_df_prev$Prevalence, na.rm = TRUE))) %>%
    as.list()
  
  selected_metrics_equal <- averages_df_equal %>%
    summarize(across(c(F1_Score, Precision, Recall, Accuracy), ~ sum(.x * averages_df_equal$Prevalence, na.rm = TRUE))) %>%
    as.list()
  }
  
  # Calculate summary statistics
  list(
    macro_summary = list(
      F1_Score_prev = selected_metrics_prev$F1_Score,
      Precision_prev = selected_metrics_prev$Precision,
      Recall_prev = selected_metrics_prev$Recall,
      Accuracy_prev = selected_metrics_prev$Accuracy,
      F1_Score_equal = selected_metrics_equal$F1_Score,
      Precision_equal = selected_metrics_equal$Precision,
      Recall_equal = selected_metrics_equal$Recall,
      Accuracy_equal = selected_metrics_equal$Accuracy
    ),
    class_summary_prev = averages_df_prev,
    class_summary_equal = averages_df_equal
  )
}

# Unadjusted performance, random, and zero baseline -----------------------
calculate_full_multi_performance <- function(ground_truth_labels, predictions, model){
  
  # 1. absolute scores (unadjusted)
  macro_multiclass_scores <- multiclass_class_metrics(ground_truth_labels, predictions)
    class_metrics_df <- macro_multiclass_scores$class_metrics
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
  random_multiclass <- random_baseline_metrics(ground_truth_labels, iterations = 100, model)
    random_macro_summary <- random_multiclass$macro_summary
    random_class_summary_prev <- random_multiclass$class_summary_prev
    random_class_summary_equal <- random_multiclass$class_summary_equal
    
  # Calculate false positives per predicted class
  confusion_matrix <- table(Predicted = predictions, Actual = ground_truth_labels)
  false_positives <- sapply(rownames(confusion_matrix), function(class) {
    # Sum all cases where we predicted this class (row) but actual (column) was different
    sum(confusion_matrix[class, colnames(confusion_matrix) != class])
  })
  false_positives_df <- data.frame(
    Class = names(false_positives),
    FalsePositives = as.numeric(false_positives)
  )
  
  # Compile results
  macro_results <- data.frame(
    Dataset = dataset_name,
    Model = model,
    Activity = "WeightedMacroAverage",
    
    Prevalence = NA,
    FalsePositives = NA,
    
    F1_Score = weighted_metrics["F1_Score"],
    Precision = weighted_metrics["Precision"],
    Recall = weighted_metrics["Recall"],
    Accuracy = weighted_metrics["Accuracy"],
    
    ZeroR_F1_Score = macro_metrics_zero$F1_Score,
    ZeroR_Precision = macro_metrics_zero$Precision,
    ZeroR_Recall = macro_metrics_zero$Recall,
    ZeroR_Accuracy = macro_metrics_zero$Accuracy,
    
    Random_F1_Score_prev = random_macro_summary$F1_Score_prev,
    Random_Precision_prev = random_macro_summary$Precision_prev,
    Random_Recall_prev = random_macro_summary$Recall_prev,
    Random_Accuracy_prev = random_macro_summary$Accuracy_prev,
    
    Random_F1_Score_equal = random_macro_summary$F1_Score_equal,
    Random_Precision_equal = random_macro_summary$Precision_equal,
    Random_Recall_equal = random_macro_summary$Recall_equal,
    Random_Accuracy_equal = random_macro_summary$Accuracy_equal
  )

  activity_results_list <- list()
  # Loop through each unique activity
  for (activity in unique(ground_truth_labels)) {
    # Create dataframe for current activity
    
    
    false_positive_activity <- with(false_positives_df, 
                                    ifelse(is.null(FalsePositives[Class == activity]) || FalsePositives[Class == activity] == 0, 
                                           0, 
                                           FalsePositives[Class == activity])
    )
    
    activity_results_list[[activity]] <- data.frame(
      Dataset = dataset_name,
      Model = model,
      Activity = activity,
      
      Prevalence = class_metrics_df$Prevalence[class_metrics_df$Class == activity],
      FalsePositives = false_positive_activity,
      
      F1_Score = class_metrics_df$F1_Score[class_metrics_df$Class == activity],
      Precision = class_metrics_df$Precision[class_metrics_df$Class == activity],
      Recall = class_metrics_df$Recall[class_metrics_df$Class == activity],
      Accuracy = class_metrics_df$Accuracy[class_metrics_df$Class == activity],
      
      Random_F1_Score_prev = random_class_summary_prev$F1_Score[random_class_summary_prev$Class == activity],
      Random_Precision_prev = random_class_summary_prev$Precision[random_class_summary_prev$Class == activity],
      Random_Recall_prev = random_class_summary_prev$Recall[random_class_summary_prev$Class == activity],
      Random_Accuracy_prev = random_class_summary_prev$Accuracy[random_class_summary_prev$Class == activity],
      
      Random_F1_Score_equal = random_class_summary_equal$F1_Score[random_class_summary_equal$Class == activity],
      Random_Precision_equal = random_class_summary_equal$Precision[random_class_summary_equal$Class == activity],
      Random_Recall_equal = random_class_summary_equal$Recall[random_class_summary_equal$Class == activity],
      Random_Accuracy_equal = random_class_summary_equal$Accuracy[random_class_summary_equal$Class == activity],
      
      ZeroR_F1_Score = class_metrics_zero$F1_Score[class_metrics_zero$Class == activity],
      ZeroR_Precision = class_metrics_zero$Precision[class_metrics_zero$Class == activity],
      ZeroR_Recall = class_metrics_zero$Recall[class_metrics_zero$Class == activity],
      ZeroR_Accuracy = class_metrics_zero$Accuracy[class_metrics_zero$Class == activity]
    )
  }
  
  # Combine all results into a single dataframe
  final_class_results <- do.call(rbind, activity_results_list)
  
  # Combine all results
  test_results <- rbind(macro_results, final_class_results)
}

