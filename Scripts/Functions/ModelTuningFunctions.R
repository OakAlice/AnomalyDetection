
#  Functions for tuning the model hyperparameters across k_fold loops --------

split_data <- function(feature_data, validation_proportion) {
  unique_ids <- unique(feature_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
  training_data <- feature_data[!feature_data$ID %in% test_ids, ]
  validation_data <- feature_data[feature_data$ID %in% test_ids, ]
  return(list(training_data = training_data, validation_data = validation_data))
}

# 1-Class Model Tuning ----------------------------------------------------
OCCModelTuning <- function(feature_data, target_activity, nu, kernel, gamma, number_trees, number_features, validation_proportion = 0.2, balance = "non_stratified_balance") {
  tryCatch({
    # Adjust kernel value
    kernel <- ifelse(kernel < 0.5, "linear", ifelse(kernel < 1.5, "radial", "polynomial"))
    
    outcomes <- list()  # List to store outcomes from multiple runs
    
    # Parallelize the loop for 3 iterations
    iterations <- 1:3
    future_outcomes <- future_lapply(iterations, function(i) {
      tryCatch({
        
        set.seed(i)
        # Split data into training and validation sets
        data_split <- split_data(feature_data, validation_proportion)
        training_data <- as.data.table(data_split$training_data)
        validation_data <- as.data.table(data_split$validation_data)
        
        # Feature selection
        # select only instances from the training condition, no access to outside information
        target_feature_data <- training_data[Activity == target_activity, ]
        selected_feature_data <- featureSelection(target_feature_data, number_trees, number_features)
        selected_feature_data <- selected_feature_data[complete.cases(selected_feature_data), ]
        
        # Train SVM model
        selected_numeric_data <- selected_feature_data[, !"Activity", with = FALSE]
        
        svm_args <- list(x = selected_numeric_data, 
                         y = NULL, 
                         type = "one-classification", 
                         nu = nu, 
                         scale = TRUE, 
                         kernel = kernel, 
                         gamma = gamma)
        single_class_SVM <- do.call(svm, svm_args)
        
        # Validate model
        # select the validation data, clean and format it
        top_features <- colnames(selected_feature_data)
        selected_validation_data <- validation_data[, .SD, .SDcols = top_features]
        
        # Balance validation data
        if (balance == "non_stratified_balance") {
          activity_count <- selected_validation_data[Activity == target_activity, .N]
          selected_validation_data <- selected_validation_data[, .SD[1:activity_count], by = Activity]
        } else if (balance == "stratified_balance") {
          activity_count <- selected_validation_data[Activity == target_activity, .N] / length(unique(selected_validation_data$Activity))
          selected_validation_data <- selected_validation_data[, .SD[sample(.N, min(.N, activity_count))], by = Activity]
        }
        
        selected_validation_data[Activity != target_activity, Activity := "Other"]
        selected_validation_data <- selected_validation_data[complete.cases(selected_validation_data), ]
        
        # Generate predictions and calculate performance
        ground_truth <- ifelse(selected_validation_data$Activity == as.character(target_activity), 1, -1)
        numeric_validation_data <- selected_validation_data[, !"Activity"]
        decision_scores <- predict(single_class_SVM, newdata = numeric_validation_data, decision.values = TRUE)
        scores <- as.numeric(attr(decision_scores, "decision.values"))
        
        results <- calculateThresholdMetrics(scores, ground_truth)
        
        # Compile results for this run
        list(
          Activity = as.character(target_activity),
          nu = as.character(nu),
          gamma = as.character(gamma),
          kernel = as.character(kernel),
          number_features = as.character(number_features),
          number_trees = as.character(number_trees),
          F1_Score = as.numeric(results$F1_score),
          top_features = as.character(top_features)
        )
      }, error = function(e) {
        message(sprintf("Error during run %d: %s", i, e$message))
        NULL
      })
    }, future.seed = TRUE)
    
    # Combine the outcomes from the parallelized runs
    model_outcomes <- rbindlist(future_outcomes, use.names = TRUE, fill = TRUE)
    avg_outcomes <- model_outcomes[, .(mean_F1 = mean(F1_Score, na.rm = TRUE)), 
                                   by = .(Activity, nu, gamma, kernel, number_features)]
    best_index <- which.max(sapply(future_outcomes, function(x) x$F1_Score))
    top_features <- future_outcomes[[best_index]]$top_features
      
    list(Score = as.numeric(avg_outcomes$mean_F1), Pred = top_features)
  }, error = function(e) {
    message("Error during model tuning: ", e$message)
    list(Score = NA, Pred = NA)
  })
}


# Binary Model Tuning -----------------------------------------------------
binaryModelTuning <- function(feature_data, target_activity, nu, kernel, gamma, number_trees, number_features, validation_proportion = 0.33) {
  tryCatch({
    # Map kernel value to type
    kernel <- ifelse(kernel < 0.5, "linear", ifelse(kernel < 1.5, "radial", "polynomial"))
    outcomes_list <- list()  # Store results for each fold
    
    # Parallelize the loop for 3 folds
    folds <- 1:3
    future_outcomes <- future_lapply(folds, function(i) {
      tryCatch({
        
        set.seed(i)
        # Split data into training and validation sets
        data_split <- split_data(feature_data, validation_proportion)
        training_data <- data_split$training_data
        validation_data <- data_split$validation_data
        
        # Binary encoding of activity
        training_data[Activity != target_activity, Activity := "Other"]
        
        # Feature selection
        selected_features <- featureSelection(training_data, number_trees, number_features)
        selected_features <- selected_features[complete.cases(selected_features), ]
        
        # Train SVM model # this time with Activity as the binary
        # account for the imbalance in the data by setting class weights
        class_weights <- table(selected_features$Activity)
        class_weights <- max(class_weights) / class_weights
        
        # Fit the SVM model with parameters for this round
        svm_model <- svm(
          x = as.matrix(selected_features[, !("Activity"), with = FALSE]),
          y = as.factor(selected_features$Activity),
          type = "C-classification",
          nu = nu,
          kernel = kernel,
          gamma = gamma,
          scale = TRUE,
          class.weights = class_weights 
        )
        
        # Validation
        top_features <- colnames(selected_features)
        validation_data <- validation_data[, ..top_features]
        validation_data <- validation_data[complete.cases(validation_data), ]
        validation_data[Activity != target_activity, Activity := "Other"]
        
        predictions <- predict(svm_model, newdata = as.matrix(validation_data[, !("Activity"), with = FALSE]))
        confusion <- table(predictions, validation_data$Activity)
        
        # Compute F1 score and then average
        metrics <- confusionMatrix(confusion)
        f1_scores <- metrics$byClass["F1"]
        f1_scores[is.na(f1_scores)] <- 0 # in case of NA, set to 0 before taking macro
        macro_f1 <- mean(f1_scores, na.rm = TRUE)
        
        # Store fold results
        data.table(
          Activity = target_activity,
          nu = nu,
          gamma = gamma,
          kernel = kernel,
          number_features = number_features,
          number_trees = number_trees,
          F1_Score = macro_f1
        )
      }, error = function(e) {
        message(sprintf("Error in fold %d: %s", i, e$message))
        NULL
      })
    }, future.seed = TRUE)
    
    # Combine the outcomes from the parallelized folds
    model_outcomes <- rbindlist(future_outcomes, use.names = TRUE, fill = TRUE)
    mean_F1 <- model_outcomes[, mean(F1_Score, na.rm = TRUE)]
    
    list(Score = mean_F1, Pred = top_features)  # Return average F1 score
  }, error = function(e) {
    message("Error in binary model tuning: ", e$message)
    return(list(Score = NA, Pred = NA))  # Return NA on error
  })
}


# Multiclass Model Tuning -------------------------------------------------
multiclassModelTuning <- function(multiclass_data, nu, kernel, gamma, number_trees, number_features, validation_proportion, loops) {
  
  # Convert kernel from numeric to kernel type
  kernel_type <- ifelse(kernel < 0.5, "linear", 
                        ifelse(kernel < 1.5, "radial", "polynomial"))
  
  # Parallelise loop over iterations
  num_loops <- 1:loops
  f1_scores <- future_lapply(num_loops, function(i) {
    
    tryCatch({
      set.seed(i)
      # Split data into training and validation sets
      data_split <- split_data(multiclass_data, validation_proportion)
      training_data <- data_split$training_data
      validation_data <- data_split$validation_data
      
      # Feature selection
      selected_feature_data <- featureSelection(training_data, number_trees, number_features)
      selected_feature_data <- na.omit(selected_feature_data)  # Clean the data
      
      # Train SVM model with class balancing
      class_weights <- table(selected_feature_data$Activity)
      class_weights <- max(class_weights) / (class_weights + 1e-6)  # Prevent division by zero
      
      svm_args <- list(
        x = as.matrix(selected_feature_data[, !("Activity"), with = FALSE]),
        y = as.factor(selected_feature_data$Activity),
        type = "C-classification",
        nu = nu,
        scale = TRUE,
        kernel = kernel_type,
        gamma = gamma,
        class.weights = class_weights
      )
      multiclass_SVM <- do.call(svm, svm_args)
      
      # Validate the model
      top_features <- colnames(selected_feature_data)
      validation_features <- validation_data[, ..top_features]
      validation_features <- na.omit(validation_features)  # Clean validation data
      numeric_validation_data <- as.matrix(validation_features[, !("Activity"), with = FALSE])
      ground_truth_labels <- validation_features$Activity
      
      # Predictions
      predictions <- predict(multiclass_SVM, newdata = numeric_validation_data)
      
      # Compute confusion matrix and F1-scores
      confusion_matrix <- table(predictions, ground_truth_labels)
      all_classes <- sort(union(rownames(confusion_matrix), colnames(confusion_matrix)))
      conf_matrix_padded <- matrix(0, nrow = length(all_classes), ncol = length(all_classes),
                                   dimnames = list(all_classes, all_classes))
      conf_matrix_padded[rownames(confusion_matrix), colnames(confusion_matrix)] <- confusion_matrix
      
      # Calculate the F1 scores using confusionMatrix function
      metrics <- confusionMatrix(conf_matrix_padded)
      f1 <- metrics$byClass[, "F1"]
      f1[is.na(f1)] <- 0  # Replace NAs with 0
      return(mean(f1))  # Return the macro F1 score from this iteration
      
    }, error = function(e) {
      message("Error during iteration ", i, ": ", e$message)
      return(NA)  # Return NA when error
    })
  }, future.seed = TRUE)
  
  # Return average F1-score
  average_macro_f1 <- mean(f1_scores, na.rm = TRUE)
  
  # Return results
  return(list(Score = average_macro_f1, Pred = top_features))
}

