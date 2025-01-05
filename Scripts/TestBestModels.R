# Testing the best models -------------------------------------------------
# load in the test data
test_feature_data <- fread(file = file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

# make a minor formatting change
test_feature_data$GeneralisedActivity <- str_to_title(test_feature_data$GeneralisedActivity)


# Dichotomous models ------------------------------------------------------
test_results <- data.frame()

for(model in c("OCC", "Binary")){
  for(behaviour in target_activities){
    # load in the model, comes in as "trained_SVM"
    load(file.path(base_path, "Output", "Models", paste0(dataset_name, "_", model, "_", behaviour, "_final_model.rda")))
    
    # extract the right variables from the model
    selected_features <- colnames(trained_SVM$call$x)
    
    # prepare the test data
    selected_test_feature_data <- test_feature_data[, .SD, .SDcols = c(selected_features, "Activity", "Time", "ID")]
    selected_test_feature_data <- na.omit(selected_test_feature_data)
    selected_test_feature_data <- as.data.table(selected_test_feature_data)
    selected_test_feature_data$Activity <- ifelse(selected_test_feature_data$Activity == behaviour, behaviour, "Other")
    ground_truth_labels <- selected_test_feature_data$Activity
    time_values <- selected_test_feature_data$Time
    ID_values <- selected_test_feature_data$ID
    numeric_test_data <- selected_test_feature_data[, !c("Activity", "Time", "ID"), with = FALSE]
    
    # this bit was important for the seal data, don't remove
    invalid_row_indices <- which(!complete.cases(numeric_test_data) |
                                   !apply(numeric_test_data, 1, function(row) all(is.finite(row))))
    
    if (length(invalid_row_indices) > 0) {
      numeric_test_data <- numeric_test_data[-invalid_row_indices, , drop = FALSE]
      ground_truth_labels <- ground_truth_labels[-invalid_row_indices]
      time_values <- time_values[-invalid_row_indices]
      ID_values <- ID_values[-invalid_row_indices]
    }
    
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
    
    # Compute performance metrics
    f1_score <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    precision_metric <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    recall_metric <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    specificity_metric <- MLmetrics::Specificity(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    accuracy_metric <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = predictions)
    balanced_accuracy_metric <- calculateBalancedAccuracy(y_true = ground_truth_labels, y_pred = predictions) # custom function stored in OtherFunctions.R
    MCC_metric <- calculateMCC(y_true = ground_truth_labels, y_pred = predictions) # custom function stored in OtherFunctions.R
    
    # Compile results for this run
    results <- data.frame(
      Dataset = as.character(dataset_name),
      Model = as.character(model),
      Activity = as.character(behaviour),
      F1_Score = as.numeric(f1_score),
      Precision = as.numeric(precision_metric),
      Recall = as.numeric(recall_metric),
      Specificity = as.numeric(specificity_metric),
      Accuracy = as.numeric(accuracy_metric),
      BalancedAccuracy = as.numeric(balanced_accuracy_metric),
      MCC = as.numeric(MCC_metric)
    )
  
    test_results <- rbind(test_results, results)
    message("test results stored")
    
    # write out the predictions for later plotting
    output <- data.table(
      "Time" = time_values,
      "ID" = ID_values,
      "Ground_truth" = ground_truth_labels, 
      "Predictions" = predictions,
      "Decision_values" = decision_values
    )
    fwrite(output, file.path(base_path, "Output", "Testing", "Predictions", paste(dataset_name, model, behaviour, "predictions.csv", sep = "_")))
    message("predictions saved")
    
  }
}

# save the results
fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_dichotomous_test_performance.csv")))


# Multi-class models ------------------------------------------------------
test_results <- data.frame()
for(behaviour_set in c("Activity", "OtherActivity", "GeneralisedActivity")){
    # load in the model, comes in as "trained_SVM"
    load(file.path(base_path, "Output", "Models", paste0(dataset_name, "_Multi_", behaviour_set, "_final_model.rda")))
    
    # extract the right variables from the model
    selected_features <- colnames(trained_SVM$call$x)
    
    # prepare the test data
    multiclass_test_data <- test_feature_data %>%
      select(-(setdiff(c("Activity", "OtherActivity", "GeneralisedActivity"), behaviour_set))) %>%
      rename("Activity" = !!sym(behaviour_set))
    
    if (behaviour_set == "GeneralisedActivity") {
      multiclass_test_data <- multiclass_test_data %>% filter(!Activity == "")
    }
    
    selected_multiclass_test_data <- multiclass_test_data[, .SD, .SDcols = c(selected_features, "Activity", "Time", "ID")]
    selected_multiclass_test_data <- na.omit(selected_multiclass_test_data)
    selected_multiclass_test_data <- as.data.table(selected_multiclass_test_data)
    
    ground_truth_labels <- selected_multiclass_test_data$Activity
    time_values <- selected_multiclass_test_data$Time
    ID_values <- selected_multiclass_test_data$ID
    numeric_test_data <- selected_multiclass_test_data[, !c("Activity", "Time", "ID"), with = FALSE]
    
    # this bit was important for the seal data, don't remove
    invalid_row_indices <- which(!complete.cases(numeric_test_data) |
                                   !apply(numeric_test_data, 1, function(row) all(is.finite(row))))
    
    if (length(invalid_row_indices) > 0) {
      print("there were some invalid indices")
      numeric_test_data <- numeric_test_data[-invalid_row_indices, , drop = FALSE]
      ground_truth_labels <- ground_truth_labels[-invalid_row_indices]
      time_values <- time_values[-invalid_row_indices]
      ID_values <- ID_values[-invalid_row_indices]
    }
    
    message("testing data prepared")
    
    # make the predictions
    predictions <- predict(trained_SVM, newdata = numeric_test_data)
    
    message("predictions made")
    
    # Ensure predictions and ground_truth are factors with the same levels
    # ground_truth_labels <- sample(selected_multiclass_test_data$Activity) # randomise order
    
    unique_classes <- sort(union(predictions, ground_truth_labels))
    predictions <- factor(predictions, levels = unique_classes)
    ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)
    
    if (length(predictions) != length(ground_truth_labels)) {
      stop("Error: Predictions and ground truth labels have different lengths.")
    }
    
    # Calculate per-class metrics and macro average
    class_metrics <- lapply(unique_classes, function(class) {
      # Convert to binary problem for this class
      binary_true <- factor(ground_truth_labels == class, levels = c(FALSE, TRUE))
      binary_pred <- factor(predictions == class, levels = c(FALSE, TRUE))
      
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
      Model = behaviour_set,
      Activity = "MacroAverage",
      F1_Score = as.numeric(macro_metrics["F1_Score"]),
      Precision = as.numeric(macro_metrics["Precision"]),
      Recall = as.numeric(macro_metrics["Recall"]),
      Accuracy = as.numeric(macro_metrics["Accuracy"])
    )

    # Add per-class results
    class_level_results <- data.frame(
      Dataset = rep(as.character(dataset_name), length(unique_classes)),
      Model = rep(behaviour_set, length(unique_classes)),
      Activity = class_metrics_df$Class,
      F1_Score = class_metrics_df$F1_Score,
      Precision = class_metrics_df$Precision,
      Recall = class_metrics_df$Recall,
      Accuracy = class_metrics_df$Accuracy
    )

    # Combine macro-average and per-class results
    test_results <- rbind(test_results, macro_results, class_level_results)
    # fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_Activity_test_performance.csv")))
    message("test results stored")
    
    # write out the predictions for later plotting
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

# save the results
fwrite(test_results, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_multi_test_performance.csv")))











