# ---------------------------------------------------------------------------
# Functions for tuning the model hyperparameters across k_fold validations
# ---------------------------------------------------------------------------

# train and validate these specific values 
perform_single_validation <- function(subset_data, validation_proportion,
                                      kernel, nu, gamma, number_trees, number_features) {
  
  # Create training and validation data
  unique_ids <- unique(subset_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
  
  validation_data <- subset_data[subset_data$ID %in% test_ids, ]
  training_data <- subset_data[!subset_data$ID %in% test_ids, ]
  
  #### Feature selection ####
  # convert data to binary
  training_data$Activity <- ifelse(training_data$Activity == as.character(targetActivity), 
                                   training_data$Activity, 
                                   "Other")
  selected_feature_data <- feature_selection(training_data, number_trees, number_features)
  
  #### Train model ####
  label_columns <- c("Activity", "Time", "ID")
  target_class_feature_data <- selected_feature_data[Activity == as.character(targetActivity),
                                                       !label_columns, with = FALSE] 

  single_class_SVM <- do.call(svm, list(target_class_feature_data, y = NULL, type = 'one-classification', 
                                            nu = nu, scale = TRUE, kernel = kernel))
  
  #### Validate model ####
  top_features <- setdiff(colnames(selected_feature_data), label_columns)
  selected_validation_data <- validation_data[, .SD, .SDcols = c("Activity", top_features)]
  selected_validation_data <- selected_validation_data[complete.cases(selected_validation_data), ]
  
  # balance the validation data
  counts <- selected_validation_data[Activity == targetActivity, .N]
  selected_validation_data[Activity != targetActivity, Activity := "Other"]
  selected_validation_data <- selected_validation_data[, .SD[1:counts], by = Activity]
  
  ground_truth_labels <- selected_validation_data[, "Activity"]
  ground_truth_labels <- ifelse(ground_truth_labels == as.character(targetActivity), 1, -1)
  
  numeric_validation_data <- selected_validation_data[, !"Activity"]
  decision_scores <- predict(single_class_SVM, newdata = numeric_validation_data, decision.values = TRUE)
  scores <- as.numeric(attr(decision_scores, "decision.values"))
  
  # Calculate AUC-ROC
  roc_curve <- roc(as.vector(ground_truth_labels), scores)
  auc_value <- auc(roc_curve)
  # plot(roc_curve)
  
  # Calculate PR-ROC
  pr_curve <- pr.curve(scores.class0 = scores[ground_truth_labels == 1],
                       scores.class1 = scores[ground_truth_labels == -1], curve = TRUE)
  pr_auc_value <- pr_curve$auc.integral
  #plot(pr_curve)
  
  # calculate threshold metrics for threshold 0.5
  metrics_0.5_threshold <- threshold_metrics(scores, ground_truth_labels, threshold = 0.5)
  
  # Create a tibble for the cross-validation result
  cross_result <- tibble(
    Activity = as.character(targetActivity),
    
    nu = as.character(nu),
    gamma = as.character(gamma),
    kernel = as.character(kernel),
    number_features = as.character(number_features),
    number_trees = as.character(number_trees),
    
    AUC_Value = as.numeric(auc_value),
    PR_AUC = as.numeric(pr_auc_value),
    Accuracy = as.numeric(metrics_0.5_threshold$Accuracy),
    Balanced_Accuracy = as.numeric(metrics_0.5_threshold$Balanced_Accuracy),
    Precision = as.numeric(metrics_0.5_threshold$Precision),
    Recall = as.numeric(metrics_0.5_threshold$Recall),
    F1_Score = as.numeric(metrics_0.5_threshold$F1_score)
  )
  
  return(cross_result)
}
