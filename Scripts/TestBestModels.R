# Testing the best models -------------------------------------------------
# load in the test data
test_feature_data <- fread(file = file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

# make a minor formatting change
test_feature_data$GeneralisedActivity <- str_to_title(test_feature_data$GeneralisedActivity)

# Dichotomous models -----------------------------------------------------
for(model in c("OCC", "Binary")){
  test_results <- data.frame()
  for(behaviour in target_activities){
    # behaviour <- target_activities[2]
    
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
    
    table(ground_truth_labels, predictions)
    
    # Compute performance metrics
    f1_score <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    precision_metric <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    recall_metric <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = predictions, positive = behaviour)
    accuracy_metric <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = predictions)
   
    # Compute baseline metrics
    zero_rate_baseline <- as.list(calculate_zero_rate_baseline(ground_truth_labels, model, behaviour))
    random_baseline <- as.list(calculate_random_baseline(ground_truth_labels, behaviour, n_iterations = 100))
    
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
      ZeroR_Accuracy = zero_rate_baseline$Accuracy,
      Random_F1 = random_baseline$F1_Score,
      Random_Precision = random_baseline$Precision,
      Random_Recall = random_baseline$Recall,
      Random_Accuracy = random_baseline$Accuracy
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
  
  if (length(predictions) != length(ground_truth_labels)) {
    stop("Error: Predictions and ground truth labels have different lengths.")
  }
  
  test_results <- calculate_full_multi_performance(ground_truth_labels, predictions)

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
