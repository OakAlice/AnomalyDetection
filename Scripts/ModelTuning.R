# ---------------------------------------------------------------------------
# Functions for tuning the model hyperparameters across k_fold validations
# ---------------------------------------------------------------------------

# train and validate these specific values 
performSingleValidation <- function(feature_data, validation_proportion,
                                    kernel, nu, gamma, number_trees, number_features) {
  tryCatch({
    #### Create training and validation data ####
    unique_ids <- unique(feature_data$ID)
    test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
    
    training_data <- feature_data[!feature_data$ID %in% test_ids, ]
    validation_data <- feature_data[feature_data$ID %in% test_ids, ]
    
  }, error = function(e) {
    message("Error in data splitting: ", e$message)
    flush.console()
    return(NULL)
  })
  
  training_data$Activity <- ifelse(training_data$Activity == as.character(target_activity), 
                                   training_data$Activity, "Other")
  tryCatch({
    #### Feature selection ####
    selected_feature_data <- featureSelection(training_data, number_trees, number_features)
  }, error = function(e) {
    message("Error during general feature selection: ", e$message)
    flush.console()
  })
  
  tryCatch({
    #### Train model ####
    target_class_feature_data <- selected_feature_data[Activity == as.character(target_activity),
                                                       !label_columns, with = FALSE]
    
    target_class_feature_data <- target_class_feature_data[complete.cases(target_class_feature_data), ]
    
    # SVM arguments for one-class classification
    svm_args <- list(
      x = target_class_feature_data,
      y = NULL,
      type = "one-classification",
      nu = nu,
      scale = TRUE,
      kernel = kernel,
      gamma = gamma
    )
    
    single_class_SVM <- do.call(svm, svm_args)
    
  }, error = function(e) {
    message("Error during model training: ", e$message)
    flush.console()
    return(NULL)
  })
  
  tryCatch({
    #### Validate model ####
    top_features <- setdiff(colnames(selected_feature_data), label_columns)
    selected_validation_data <- validation_data[, .SD, .SDcols = c("Activity", top_features)]
    
    # Balance validation data
    counts <- selected_validation_data[Activity == target_activity, .N]
    selected_validation_data[Activity != target_activity, Activity := "Other"]
    selected_validation_data <- selected_validation_data[, .SD[1:counts], by = Activity]
    selected_validation_data <- selected_validation_data[complete.cases(selected_validation_data), ]
    
    # Ground truth labels
    ground_truth_labels <- selected_validation_data[, "Activity"]
    ground_truth_labels <- ifelse(ground_truth_labels == as.character(target_activity), 1, -1)
    
    numeric_validation_data <- selected_validation_data[, !"Activity"]
    decision_scores <- predict(single_class_SVM, newdata = numeric_validation_data, decision.values = TRUE)
    scores <- as.numeric(attr(decision_scores, "decision.values"))
    
    results <- calculatePerformance(scores, ground_truth_labels)
    
  }, error = function(e) {
    message("Error during model validation: ", e$message)
    flush.console()
    return(NULL)
  })
  
  tryCatch({
    #### Compile results ####
    cross_result <- tibble(
      Activity = as.character(target_activity),
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
    
  }, error = function(e) {
    message("Error during result compilation: ", e$message)
    flush.console()
    return(NULL)
  })
}


calculatePerformance <- function(scores, ground_truth_labels){
  
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
  metrics_0.5_threshold <- calculateThresholdMetrics(scores, ground_truth_labels)
  
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
modelTuning <- function(feature_data, nu, kernel, gamma, number_trees, number_features) {
  tryCatch({
    # Perform a single validation three times
    outcomes_list <- list()
    
    # Run the validation function 3 times
    for (i in 1:3) {
      tryCatch({
        result <- performSingleValidation(
          feature_data, 
          validation_proportion, 
          kernel = kernel,
          nu = nu,
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
        outcomes_list[[i]] <- result  # Store all the results
        
      }, error = function(e) {
        message("Error in performSingleValidation during iteration ", i, ": ", e$message)
        flush.console() 
        return(NULL)  # Skip if error
      })
    }
    
    # Combine the outcomes into a single data.table
    tryCatch({
      model_outcomes <- rbindlist(outcomes_list, use.names = TRUE, fill = TRUE)
    }, error = function(e) {
      message("Error in rbindlist: ", e$message)
      flush.console()
      return(list(Score = NA, Pred = NA))
    })
    
    # Calculate the mean and standard deviation of PR_AUC
      average_model_outcomes <- model_outcomes[, .(
        mean_PR_AUC = mean(PR_AUC, na.rm = TRUE),
        sd_PR_AUC = sd(PR_AUC, na.rm = TRUE)
      ), by = .(Activity, nu, gamma, kernel, number_features)]
    
    # Extract the mean PR_AUC for optimization
    PR_AUC <- as.numeric(average_model_outcomes$mean_PR_AUC)
    
    # Return a named list with 'Score' and 'Pred'
    return(list(
      Score = PR_AUC,  # Metric to be maximized
      Pred = NA  # Placeholder for predictions which I dont have here
    ))
    
  }, error = function(e) {
    message("Error encountered during modelTuning: ", e$message)
    flush.console() 
    return(list(Score = NA, Pred = NA))  # Return NA for the iteration
  })
}

