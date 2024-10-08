#---------------------------------------------------------------------------
# Baseline performance of multi-class SVM                               ####
#---------------------------------------------------------------------------

baselineMultiClass <- function(training_data, testing_data, number_trees, number_features){

  selected_feature_data <- featureSelection(training_data, number_trees, number_features)
  selected_feature_data <- selected_feature_data[, !c("Time", "ID"), with = FALSE]
  selected_feature_data$Activity <- as.factor(selected_feature_data$Activity)
  selected_feature_data <- selected_feature_data[complete.cases(selected_feature_data), ]
 
  
  # Tune multiclass SVM parameters ### TODO
  
  # Train multiclass SVM with chosen parameters 
  kernel <- "radial"
  cost <- 0.092827
  gamma <- 0.08586
  
  svm_model <- svm(Activity ~ ., 
                   data = selected_feature_data, 
                   type = 'C-classification',   
                   kernel = kernel,           
                   cost = cost,           
                   gamma = gamma)
  
  
  # validate
  top_features <- colnames(selected_feature_data)
  testing_data <- testing_data[, ..top_features]
  testing_data <- testing_data[complete.cases(testing_data), ]
  numeric_testing_data <- testing_data %>% select(!"Activity")
  
  predictions <- predict(svm_model, newdata = numeric_testing_data)
  ground_truth_labels <- testing_data$Activity
  
  confusion_matrix <- table(predictions, ground_truth_labels)

  # Add row and column names
  rownames(confusion_matrix) <- c("Bowing", "Carrying object", "Drinking", "Eating", "Galloping", "Jumping", "Lying chest", "Pacing", "Panting", "Playing", "Shaking", "Sitting", "Sniffing", "Standing", "Trotting", "Tugging", "Walking")
  colnames(confusion_matrix) <- rownames(confusion_matrix)
  
  # Calculate performance metrics
  confusion_mtx <- confusionMatrix(confusion_matrix)
  
  # Extract precision, recall, and F1-score
  precision <- confusion_mtx$byClass[, "Precision"]
  recall <- confusion_mtx$byClass[, "Recall"]
  f1 <- confusion_mtx$byClass[, "F1"]
  accuracy <- confusion_mtx$byClass[, "Balanced Accuracy"]
  
  # Calculate macro-averaged precision, recall, and F1-score
  macro_precision <- mean(precision, na.rm = TRUE)
  macro_recall <- mean(recall, na.rm = TRUE)
  macro_f1 <- mean(f1, na.rm = TRUE)
  macro_accuracy <- mean(accuracy, na.rm = TRUE)
  
  return(list(F1_score = macro_f1,
              Precision = macro_precision,
              Recall = macro_recall,
              Accuracy = macro_accuracy
              ))
}
