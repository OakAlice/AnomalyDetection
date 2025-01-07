# Random performance baseline for dichotomous models ------------------------
calculate_random_baseline <- function(ground_truth_labels, activity, n_iterations = 100) {
  # Get class proportions from ground truth
  class_props <- table(ground_truth_labels) / length(ground_truth_labels)
  
  # Store results for each iteration
  iteration_metrics <- matrix(0, nrow = n_iterations, ncol = 4)
  colnames(iteration_metrics) <- c("F1_Score", "Precision", "Recall", "Accuracy")
  
  for(i in 1:n_iterations) {
    # Generate random predictions maintaining class proportions
    random_preds <- sample(
      x = levels(as.factor(ground_truth_labels)),
      size = length(ground_truth_labels),
      prob = class_props,
      replace = TRUE
    )
    random_preds <- factor(random_preds, levels = levels(ground_truth_labels))
    
    # Calculate metrics for this iteration
    iteration_metrics[i, "F1_Score"] <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = random_preds,
                                                            positive = activity
    )
    iteration_metrics[i, "Precision"] <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = random_preds,
                                                              positive = activity
    )
    iteration_metrics[i, "Recall"] <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = random_preds,
                                                        positive = activity
    )
    iteration_metrics[i, "Accuracy"] <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = random_preds
    )
  }
  
  # Return mean metrics across iterations
  colMeans(iteration_metrics, na.rm = TRUE)
}


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


# Function for getting the per class and macro metrics --------------------
multiclass_class_metrics <- function(ground_truth_labels, predictions) { 
  # Ensure predictions and ground_truth are factors with the same levels
  unique_classes <- sort(union(predictions, ground_truth_labels))
  predictions <- factor(predictions, levels = unique_classes)
  ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)
  
  # Calculate per-class metrics and macro average
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
  
  # Calculate macro averages
  macro_metrics <- colMeans(replace(class_metrics_df[, c("F1_Score", "Precision", "Recall", "Accuracy")], 
                                    is.na(class_metrics_df[, c("F1_Score", "Precision", "Recall", "Accuracy")]), 
                                    0))
  macro_metrics <- as.list(macro_metrics)
  
  return(list(macro_metrics = macro_metrics,
              class_metrics = class_metrics))
}







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


# For multiclass metrics --------------------------------------------------
random_baseline_metrics <- function(ground_truth_labels, iterations = 100) {
  # Convert ground truth to factor and get class information
  ground_truth_labels <- factor(ground_truth_labels)
  class_levels <- levels(ground_truth_labels)
  class_props <- prop.table(table(ground_truth_labels))
  n_classes <- length(class_levels)
  n_samples <- length(ground_truth_labels)
  
  # Pre-allocate matrices for storing results
  macro_metrics <- matrix(0, nrow = iterations, ncol = 4,
                          dimnames = list(NULL, c("F1_Score", "Precision", "Recall", "Accuracy")))
  
  class_metrics <- array(0, dim = c(iterations, n_classes, 4),
                         dimnames = list(NULL, class_levels, 
                                         c("F1_Score", "Precision", "Recall", "Accuracy")))
  
  # Run iterations
  for (i in 1:iterations) {
    # Generate random predictions
    random_preds <- factor(
      sample(class_levels, size = n_samples, prob = class_props, replace = TRUE),
      levels = class_levels
    )
    
    # Get confusion matrix
    conf_matrix <- table(Predicted = random_preds, Actual = ground_truth_labels)
    
    # Calculate metrics for each class
    for (j in seq_len(n_classes)) {
      TP <- conf_matrix[j, j]
      FP <- sum(conf_matrix[j, ]) - TP
      FN <- sum(conf_matrix[, j]) - TP
      TN <- sum(conf_matrix) - (TP + FP + FN)
      
      # Calculate metrics with protection against division by zero
      precision <- if (TP + FP == 0) 0 else TP / (TP + FP)
      recall <- if (TP + FN == 0) 0 else TP / (TP + FN)
      f1 <- if (precision + recall == 0) 0 else 2 * (precision * recall) / (precision + recall)
      accuracy <- (TP + TN) / sum(conf_matrix)
      
      # Store class metrics
      class_metrics[i, j, ] <- c(f1, precision, recall, accuracy)
    }
    
    # Calculate and store macro metrics
    macro_metrics[i, ] <- colMeans(class_metrics[i, , ])
  }
  
  # Calculate summary statistics
  list(
    macro_summary = list(
      averages = colMeans(macro_metrics),
      std_devs = apply(macro_metrics, 2, sd)
    ),
    class_summary = list(
      averages = apply(class_metrics, c(2, 3), mean),
      std_devs = apply(class_metrics, c(2, 3), sd)
    ),
    iterations = iterations
  )
}
