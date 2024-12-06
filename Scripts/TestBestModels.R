# Testing highest performing hyperparmeters on the test set -----------------

# load in the best parameters (presuming you made them into a csv)
occ_hyperparam_file <- file.path(base_path, "Output", paste0(dataset_name, "_OCC_hyperparmaters.csv"))
binary_hyperparam_file <- file.path(base_path, "Output", paste0(dataset_name, "_Binary_hyperparmaters.csv"))
multi_hyperparam_file <- file.path(base_path, "Output", paste0(dataset_name, "_Multi_hyperparmaters.csv"))


# OCC models ----------------------------------------------------------------
for (i in seq_len(nrow(occ_hyperparam_file))) {
  parameter_row <- occ_hyperparam_file[i, ]
  
  # Define paths
  model_path <- file.path(base_path, "Output", "Models", paste0(parameter_row$data_name, "_", parameter_row$activity, "_final_model.rda"))
  results_path <- file.path(base_path, "Output", paste0(parameter_row$data_name, "_", parameter_row$activity, "_OCC_results.csv"))
  predictions_path <- file.path(base_path, "Output", "Predictions", paste0(parameter_row$data_name, "_", parameter_row$activity, "_OCC_predictions.csv"))
  
  # Check if model exists
  if (file.exists(model_path)) {
    print("Model already exists, loading...")
    load(model_path)
    top_features <- names(optimal_single_class_SVM$x.scale$`scaled:center`)
  } else {
    # Load and preprocess training data
    training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv"))) %>%
      select(-c("OtherActivity", "GeneralisedActivity")) %>%
      mutate(Activity = ifelse(Activity == parameter_row$activity, parameter_row$activity, "Other"))
    
    # Feature selection
    selected_features <- featureSelection(training_data, parameter_row$number_trees, parameter_row$number_features)
    target_data <- selected_features[Activity == parameter_row$activity, !c("Activity", "Time", "ID"), with = FALSE]
    
    # Train SVM
    optimal_single_class_SVM <- do.call(
      svm,
      list(
        x = target_data,
        y = NULL,
        type = "one-classification",
        nu = parameter_row$nu,
        scale = TRUE,
        kernel = parameter_row$kernel,
        gamma = parameter_row$gamma
      )
    )
    
    # Save model and extract top features
    save(optimal_single_class_SVM, file = model_path)
    top_features <- colnames(target_data)
  }
  
  # Load and preprocess testing data
  test_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv"))) %>%
    select(c("Activity", "Time", "ID", top_features)) %>%
    na.omit()
  
  # Balance testing data
  if (balance == "non_stratified_balance") {
    activity_count <- test_data[Activity == parameter_row$activity, .N]
    test_data[Activity != parameter_row$activity, Activity := "Other"]
    test_data <- test_data[, .SD[1:activity_count], by = Activity]
  } else if (balance == "stratified_balance") {
    activity_count <- test_data[Activity == parameter_row$activity, .N] / length(unique(test_data$Activity))
    test_data <- test_data[, .SD[sample(.N, min(.N, activity_count))], by = Activity]
  }
  
  test_data <- test_data %>% arrange(ID, Time)
  
  # Extract ground truth labels
  ground_truth_labels <- ifelse(test_data$Activity == parameter_row$activity, 1, -1)
  numeric_features <- test_data[, !c("Activity", "Time", "ID"), with = FALSE]
  
  # Predict decision scores
  tryCatch({
    decision_scores <- predict(optimal_single_class_SVM, newdata = numeric_features, decision.values = TRUE)
    predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))
  }, error = function(e) {
    stop(paste("Error in prediction:", e$message))
  })
  
  # Save results if not already present
  if (!file.exists(results_path)) {
    results <- calculatePerformance(predicted_scores, ground_truth_labels) %>%
      as.data.frame() %>%
      mutate(data_name = parameter_row$data_name, activity = parameter_row$activity)
    fwrite(results, results_path)
  } else {
    print("Results already exist.")
  }
  
  # Save predictions if not already present
  if (!file.exists(predictions_path)) {
    predicted_classes <- ifelse(predicted_scores > results$threshold, 1, -1)
    predictions <- cbind(
      predicted_classes,
      test_data[, .SD, .SDcols = c("Activity", "ID", "Time")],
      GroundTruth = ground_truth_labels
    )
    fwrite(predictions, predictions_path)
  } else {
    print("Predictions already exist.")
  }
}

# Testing binary models ---------------------------------------------------
for (i in seq_len(nrow(binary_hyperparam_file))) {
  parameter_row <- binary_hyperparam_file[i, ]
  
  # Paths for saving model and results
  model_path <- file.path(base_path, "Output", "Models", paste0(parameter_row$data_name, "_", parameter_row$activity, "_binary_final_model.rda"))
  predictions_path <- file.path(base_path, "Output", "Predictions", paste0(parameter_row$data_name, "_", parameter_row$activity, "_binary_predictions.csv"))
  
  # Load and preprocess training data
  training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv"))) %>%
    select(-c("OtherActivity", "GeneralisedActivity")) %>%
    mutate(Activity = ifelse(Activity == parameter_row$activity, parameter_row$activity, "Other"))
  
  # Feature selection and filtering
  selected_feature_data <- featureSelection(training_data, parameter_row$number_trees, parameter_row$number_features) %>% na.omit()
  top_features <- setdiff(colnames(selected_feature_data), c("Activity"))
  
  # Train SVM model
  binary_SVM <- svm(
    x = as.matrix(selected_feature_data[, top_features, with = FALSE]),
    y = as.factor(selected_feature_data$Activity),
    type = "C-classification",
    nu = parameter_row$nu,
    scale = TRUE,
    kernel = parameter_row$kernel,
    gamma = parameter_row$gamma
  )
  
  # Save trained model
  save(binary_SVM, file = model_path)
  
  # Load and preprocess testing data
  test_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv"))) %>%
    select(c("Activity", "Time", "ID", top_features)) %>%
    na.omit()
  
  # Balance testing data depending on categroy
  if (balance == "non_stratified_balance") {
    activity_count <- test_data[Activity == parameter_row$activity, .N]
    test_data[Activity != parameter_row$activity, Activity := "Other"]
    test_data <- test_data[, .SD[1:activity_count], by = Activity]
  } else if (balance == "stratified_balance") {
    activity_count <- test_data[Activity == parameter_row$activity, .N] / length(unique(test_data$Activity))
    test_data <- test_data[, .SD[sample(.N, min(.N, activity_count))], by = Activity]
  }
  
  # Extract numeric features and ground truth labels
  numeric_features <- as.matrix(test_data[, !c("Activity", "Time", "ID"), with = FALSE])
  ground_truth_labels <- ifelse(test_data$Activity == parameter_row$activity, 1, -1)
  
  # Predict decision scores and classify
  decision_scores <- predict(binary_SVM, newdata = numeric_features, decision.values = TRUE)
  predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))
  predicted_classes <- ifelse(predicted_scores > 0, 1, -1)
  
  # Save results if not already in folder
  if (!file.exists(results_path)) {
    results <- calculatePerformance(predicted_scores, ground_truth_labels) %>%
      as.data.frame() %>%
      mutate(data_name = parameter_row$data_name, activity = parameter_row$activity)
    fwrite(results, results_path)
  } else {
    print("Results already exist")
  }
  
  # Save predictions if not already present
  if (!file.exists(predictions_path)) {
    predicted_classes <- ifelse(predicted_scores > results$threshold, 1, -1)
    predictions <- cbind(
      predicted_classes,
      test_data[, .SD, .SDcols = c("Activity", "ID", "Time")],
      GroundTruth = ground_truth_labels
    )
    fwrite(predictions, predictions_path)
  } else {
    print("Predictions already saved previously")
  }
}


# Multi-class models ------------------------------------------------------
for (i in seq_len(nrow(multi_hyperparam_file))) {
  parameter_row <- multi_hyperparam_file[i, ]
  
  # Paths for saving model and results
  model_path <- file.path(base_path, "Output", "Models", paste0(parameter_row$data_name, "_", parameter_row$activity, "_multi_model.rds"))
  results_path <- file.path(base_path, "Output", paste0(parameter_row$data_name, "_", parameter_row$activity, "_multi_results.csv"))
  predictions_path <- file.path(base_path, "Output", "Predictions", paste0(parameter_row$data_name, "_", parameter_row$activity, "_multi_predictions.csv"))
  
  # Load training and testing data
  training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv")))
  testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv")))
  
  # Update and clean training and testing feature data
  training_feature_data <- update_feature_data(training_data, parameter_row$activity) %>% filter(Activity != "")
  testing_feature_data <- update_feature_data(testing_data, parameter_row$activity) %>% filter(Activity != "")
  
  # Feature selection
  selected_feature_data <- featureSelection(training_feature_data, parameter_row$number_trees, parameter_row$number_features) %>%
    select(-c("Time", "ID")) %>%
    na.omit() %>%
    mutate(Activity = as.factor(Activity))
  
  # Train SVM model
  svm_model <- svm(
    Activity ~ .,
    data = selected_feature_data,
    type = "C-classification",
    kernel = parameter_row$kernel,
    gamma = parameter_row$gamma
  )
  
  # Save trained model
  saveRDS(svm_model, file = model_path)
  
  # Prepare testing data
  top_features <- colnames(selected_feature_data)
  testing_feature_data <- testing_feature_data[, ..top_features]
  testing_feature_data <- testing_feature_data[complete.cases(testing_feature_data), ]
  
  numeric_testing_data <- testing_feature_data %>% select(!"Activity")
  ground_truth_labels <- testing_feature_data$Activity
  
  # Predictions
  predictions <- predict(svm_model, newdata = numeric_testing_data)
  confusion_matrix <- table(predictions, ground_truth_labels)
  
  # Ensure confusion matrix dimensions match all classes
  all_classes <- sort(union(colnames(confusion_matrix), rownames(confusion_matrix)))
  conf_matrix_padded <- matrix(0,
                               nrow = length(all_classes),
                               ncol = length(all_classes),
                               dimnames = list(all_classes, all_classes))
  conf_matrix_padded[rownames(confusion_matrix), colnames(confusion_matrix)] <- confusion_matrix
  
  # Calculate performance metrics
  confusion_mtx <- confusionMatrix(conf_matrix_padded)
  precision <- confusion_mtx$byClass[, "Precision"]
  recall <- confusion_mtx$byClass[, "Recall"]
  f1 <- confusion_mtx$byClass[, "F1"]
  accuracy <- confusion_mtx$byClass[, "Balanced Accuracy"]
  
  macro_metrics <- data.frame(
    Precision = mean(precision, na.rm = TRUE),
    Recall = mean(recall, na.rm = TRUE),
    F1 = mean(f1, na.rm = TRUE),
    Accuracy = mean(accuracy, na.rm = TRUE)
  )
  
  # Save performance results
  results <- cbind(
    data_name = parameter_row$data_name,
    activity = parameter_row$activity,
    macro_metrics
  )
  fwrite(results, results_path)
  
  # Save predictions for further analysis
  outcome_classifications <- cbind(
    Predicted = predictions,
    GroundTruth = ground_truth_labels,
    testing_feature_data[, .SD, .SDcols = c("Activity", "ID", "Time")]
  )
  fwrite(outcome_classifications, predictions_path)
}
