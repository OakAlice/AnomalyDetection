# ---------------------------------------------------------------------------
# Functions for tuning the model hyperparameters across k_fold validations
# ---------------------------------------------------------------------------

# train and validate these specific values 
perform_single_validation <- function(subset_data, validation_proportion,
                                      kernel, nu, gamma, number_trees, number_features) {
  # convert the kernel
  kernel <- ifelse(kernel == 0, "linear", "radial")
  
  
  # Create training and validation data
  unique_ids <- unique(subset_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
  training_data <- subset_data[!subset_data$ID %in% test_ids, ]
  validation_data <- subset_data[subset_data$ID %in% test_ids, ]
  
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
  
  # apply the ground truth labels
  ground_truth_labels <- selected_validation_data[, "Activity"]
  ground_truth_labels <- ifelse(ground_truth_labels == as.character(targetActivity), 1, -1)
  
  numeric_validation_data <- selected_validation_data[, !"Activity"]
  decision_scores <- predict(single_class_SVM, newdata = numeric_validation_data, decision.values = TRUE)
  scores <- as.numeric(attr(decision_scores, "decision.values"))
  
  results <- get_performance(scores, ground_truth_labels)
    
  # Create a tibble for the cross-validation result
  cross_result <- tibble(
    Activity = as.character(targetActivity),
    
    nu = as.character(nu),
    gamma = as.character(gamma),
    kernel = as.character(kernel),
    number_features = as.character(number_features),
    number_trees = as.character(number_trees),
    
    AUC_Value = as.numeric(results$auc_value),
    PR_AUC = as.numeric(results$pr_auc_value),
    Accuracy = as.numeric(results$Accuracy),
    Balanced_Accuracy = as.numeric(results$Balanced_Accuracy),
    Precision = as.numeric(results$Precision),
    Recall = as.numeric(results$Recall),
    F1_Score = as.numeric(results$F1_score)
  )
  
  return(cross_result)
}



get_performance <- function(scores, ground_truth_labels){
  
  auc_value <- NA
  pr_auc_value <- NA
  
  if (length(levels(as.factor(ground_truth_labels))) == 2){
  # Calculate AUC-ROC
  roc_curve <- roc(as.vector(ground_truth_labels), scores)
  auc_value <- auc(roc_curve)
  # plot(roc_curve)
  
  # Calculate PR-ROC
  pr_curve <- pr.curve(scores.class0 = scores[ground_truth_labels == 1],
                       scores.class1 = scores[ground_truth_labels == -1], curve = TRUE)
  pr_auc_value <- pr_curve$auc.integral
  #plot(pr_curve)
  
  }
  # calculate threshold metrics
  metrics_0.5_threshold <- threshold_metrics(scores, ground_truth_labels)
  
  return(list(auc_value = auc_value,
              pr_auc_value = pr_auc_value,
              threshold = metrics_0.5_threshold$threshold,
              TP = metrics_0.5_threshold$TP,
              TN = metrics_0.5_threshold$TN,
              FP = metrics_0.5_threshold$FP,
              FN = metrics_0.5_threshold$FN,
              F1_score = metrics_0.5_threshold$F1_score,
              Precision = metrics_0.5_threshold$Precision,
              Recall = metrics_0.5_threshold$Recall,
              Recall_neg = metrics_0.5_threshold$Recall_neg,
              Accuracy = metrics_0.5_threshold$Accuracy,
              Balanced_Accuracy = metrics_0.5_threshold$Balanced_Accuracy
              ))
}

# run for each hyperparameter set
model_train_and_validate <- function(nu, kernel, gamma, number_trees, number_features) {
  
  # Perform a single validation three times
  outcomes_list <- list()
  
  # Run the validation function 3 times (you can adjust this if needed)
  for (i in 1:3) {
    result <- perform_single_validation(
      subset_data, 
      validation_proportion, 
      kernel = kernel,
      nu = nu,
      gamma = gamma,
      number_trees = number_trees,
      number_features = number_features
    )
    
    outcomes_list[[i]] <- result  # Store all of the results
  }
  # Combine the outcomes into a single data.table
  model_outcomes <- rbindlist(outcomes_list)
  
  # Calculate the mean and standard deviation of PR_AUC
  average_model_outcomes <- model_outcomes[, .(
    mean_PR_AUC = mean(PR_AUC, na.rm = TRUE),
    sd_PR_AUC = sd(PR_AUC, na.rm = TRUE)
  ), by = .(Activity, nu, gamma, kernel, number_features)]
  
  # Extract the mean PR_AUC for optimization
  PR_AUC <- as.numeric(average_model_outcomes$mean_PR_AUC)
  
  # Extract predictions (I dont want this but function requires it)
  predictions <- NA
  
  # Return a named list with 'Score' and 'Pred'
  return(list(
    Score = PR_AUC,  # Metric to be maximized
    Pred = predictions  # Validation predictions for ensembling/stacking
  ))
}