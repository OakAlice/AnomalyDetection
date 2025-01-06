# Testing the best models -------------------------------------------------
# load in the test data
test_feature_data <- fread(file = file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

# make a minor formatting change
test_feature_data$GeneralisedActivity <- str_to_title(test_feature_data$GeneralisedActivity)


# Dichotomous models -----------------------------------------------------
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


for(model in c("OCC", "Binary")){
  test_results <- data.frame()
  for(behaviour in target_activities){
    # load in the model, comes in as "trained_SVM"
    load(file.path(base_path, "Output", "Models", paste0(dataset_name, "_", model, "_", behaviour, "_final_model.rda")))
    
    # extract the right variables from the model
    selected_features <- colnames(trained_SVM$call$x)
    
    # prepare the test data
    test_data <- prepare_test_data(test_feature_data, selected_features, behaviour = behaviour)
    numeric_test_data <- as.data.frame(test_data$numeric_data)
    ground_truth_labels <- test_data$ground_truth_labels
    time_values <- test_data$time_values
    ID_values <- test_data$ID_values
    
    message("testing data prepared")
    
    # make the predictions with reported distance from hyperplane
    predictions <- predict(trained_SVM, newdata = numeric_test_data, decision.values = TRUE)
    decision_values <- attr(predictions, "decision.values")
    
    message("predictions made")
    
    if (model == "OCC") {
      predictions <- ifelse(predictions == FALSE, "Other", behaviour)
    }
    
    # Ensure predictions and ground_truth are factors with the same levels
    unique_classes <- sort(union(predictions, ground_truth_labels))
    predictions <- factor(predictions, levels = unique_classes)
    ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)
    
    if (length(predictions) != length(ground_truth_labels)) {
      stop("Error: Predictions and ground truth labels have different lengths.")
    }

    zero_rate_baseline <- as.list(calculate_zero_rate_baseline(ground_truth_labels, model, behaviour))
    
    # Compute performance metrics
    f1_score <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    precision_metric <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    recall_metric <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    accuracy_metric <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = predictions)
    
    # Compile results for this run
    results <- data.frame(
      Dataset = as.character(dataset_name),
      Model = as.character(model),
      Activity = as.character(behaviour),
      F1_Score = as.numeric(f1_score),
      Precision = as.numeric(precision_metric),
      Recall = as.numeric(recall_metric),
      Accuracy = as.numeric(accuracy_metric),
      ZeroR_F1 = zero_rate_baseline$F1_Score,
      ZeroR_Precision = zero_rate_baseline$Precision,
      ZeroR_Recall = zero_rate_baseline$Recall,
      ZeroR_Accuracy = zero_rate_baseline$Accuracy
    )
  
    test_results <- rbind(test_results, results)
    message("test results stored")
    
    # write out the predictions for later plotting
    output <- data.table(
      "Time" = time_values,
      "ID" = ID_values,
      "Ground_truth" = ground_truth_labels, 
      "Predictions" = predictions,
      "Decision_values" = as.vector(decision_values)
    )
    fwrite(output, file.path(base_path, "Output", "Testing", "Predictions", paste(dataset_name, model, behaviour, "predictions.csv", sep = "_")))
    message("predictions saved")
    
  }
  # save the results
  fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", model, "_test_performance.csv")))
}



# Multi-class models ------------------------------------------------------

for(behaviour_set in c("Activity", "OtherActivity", "GeneralisedActivity")){
  test_results <- data.frame()
  
  # load in the model, comes in as "trained_SVM"
  load(file.path(base_path, "Output", "Models", paste0(dataset_name, "_Multi_", behaviour_set, "_final_model.rda")))
  
  # extract the right variables from the model
  selected_features <- colnames(trained_SVM$call$x)
  
  # prepare the test data
  multiclass_test_data <- update_feature_data(test_feature_data, behaviour_set)
  if (behaviour_set == "GeneralisedActivity") {
    multiclass_test_data <- multiclass_test_data %>% filter(!Activity == "")
  }
  
  multiclass_test_data <- prepare_test_data(multiclass_test_data, selected_features, behaviour = NULL)
  numeric_test_data <- as.data.frame(multiclass_test_data$numeric_data)
  ground_truth_labels <- multiclass_test_data$ground_truth_labels
  time_values <- multiclass_test_data$time_values
  ID_values <- multiclass_test_data$ID_values
  
  message("testing data prepared")
  
  # make predictions
  predictions <- predict(trained_SVM, newdata = numeric_test_data)
  message("predictions made")
  
  # Ensure predictions and ground_truth are factors with the same levels
  unique_classes <- sort(union(predictions, ground_truth_labels))
  predictions <- factor(predictions, levels = unique_classes)
  ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)
  
  if (length(predictions) != length(ground_truth_labels)) {
    stop("Error: Predictions and ground truth labels have different lengths.")
  }
  
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
  
  # Get random baseline metrics
  random_baseline <- calculate_zero_rate_baseline(ground_truth_labels, model_type = "Multi", target_class = NULL)
  
  # Compile results
  macro_results <- data.frame(
    Dataset = dataset_name,
    Model = behaviour_set,
    Activity = "MacroAverage",
    F1_Score = macro_metrics["F1_Score"],
    Precision = macro_metrics["Precision"],
    Recall = macro_metrics["Recall"],
    Accuracy = macro_metrics["Accuracy"]
  )
  
  class_level_results <- data.frame(
    Dataset = rep(dataset_name, length(unique_classes)),
    Model = rep(behaviour_set, length(unique_classes)),
    Activity = class_metrics_df$Class,
    F1_Score = class_metrics_df$F1_Score,
    Precision = class_metrics_df$Precision,
    Recall = class_metrics_df$Recall,
    Accuracy = class_metrics_df$Accuracy
  )
  
  # Add random baseline results
  random_results <- data.frame(
    Dataset = dataset_name,
    Model = paste0(behaviour_set, "_Random"),
    Activity = "ZeroR_MacroAverage",
    Random_F1_Score = random_baseline["F1_Score"],
    Random_Precision = random_baseline["Precision"],
    Random_Recall = random_baseline["Recall"],
    Random_Accuracy = random_baseline["Accuracy"]
  )
  
  # Combine all results
  test_results <- rbind(test_results, random_results, macro_results, class_level_results)
  fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_Multi_", behaviour_set, "_test_performance.csv")))
  message("test results stored")
  
  # Save predictions
  output <- data.table(
    "Time" = time_values,
    "ID" = ID_values,
    "Ground_truth" = ground_truth_labels, 
    "Predictions" = predictions
  )
  
  if (nrow(output) > 0) {
    fwrite(output, file.path(base_path, "Output", "Testing", "Predictions", paste(dataset_name, behaviour_set, "predictions.csv", sep = "_")))
    message("predictions saved")
  } else {
    message("No predictions to save.")
  }
}
