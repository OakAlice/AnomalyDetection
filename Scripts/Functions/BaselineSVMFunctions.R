#---------------------------------------------------------------------------
# Baseline performance of multi-class SVM                               ####
#---------------------------------------------------------------------------

baselineMultiClass <- function(dataset_name, condition, training_data, testing_data, number_trees, number_features, kernel, gamma){

  selected_feature_data <- featureSelection(training_data, number_trees, number_features)
  selected_feature_data <- selected_feature_data[, !c("Time", "ID"), with = FALSE]
  selected_feature_data$Activity <- as.factor(selected_feature_data$Activity)
  selected_feature_data <- selected_feature_data[complete.cases(selected_feature_data), ]
 
  svm_model <- svm(Activity ~ ., 
                   data = selected_feature_data, 
                   type = 'C-classification',   
                   kernel = kernel,           
                   gamma = gamma)
  
  # save it
  saveRDS(svm_model, file = file.path(base_path, "Output", "Models", paste0(dataset_name, "_", condition, "_multi_model.rds")))
  
  # validate
  top_features <- colnames(selected_feature_data)
  testing_data <- testing_data[, ..top_features]
  testing_data <- testing_data[complete.cases(testing_data), ]
  numeric_testing_data <- testing_data %>% select(!"Activity")
  
  predictions <- predict(svm_model, newdata = numeric_testing_data)
  ground_truth_labels <- testing_data$Activity
  
  confusion_matrix <- table(predictions, ground_truth_labels)
  
  # soemtimes the confusion matrix isn't equal dimensions
    all_classes <- sort(union(colnames(confusion_matrix), rownames(confusion_matrix)))
    conf_matrix_padded <- matrix(0, 
                                 nrow = length(all_classes), 
                                 ncol = length(all_classes),
                                 dimnames = list(all_classes, all_classes))
    conf_matrix_padded[rownames(confusion_matrix), colnames(confusion_matrix)] <- confusion_matrix
    # Calculate performance metrics
    confusion_mtx <- confusionMatrix(conf_matrix_padded)
  
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
