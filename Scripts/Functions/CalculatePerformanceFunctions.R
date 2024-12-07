# Some other functions for confusion matrices -----------------------------

calculatePerformance <- function(scores, ground_truth_labels){
  
  auc_value <- NA
  pr_auc_value <- NA
  
  if (length(levels(as.factor(ground_truth_labels))) == 2){
    # Calculate AUC-ROC
    roc_curve <- roc(as.vector(ground_truth_labels), scores)
    auc_value <- as.numeric(auc(roc_curve))
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
              Accuracy = metrics_0.5_threshold$Accuracy,
              Balanced_Accuracy = metrics_0.5_threshold$Balanced_Accuracy
  ))
}




# threshold metrics
calculateThresholdMetrics <- function(scores, ground_truth_labels) {
  threshold_options <- seq(0, 1, by = 0.01)
  
  # Initialize a dataframe to store results for each threshold
  results <- data.frame(threshold = numeric(),
                        F1_score = numeric(),
                        Precision = numeric(),
                        Recall = numeric(),
                        Accuracy = numeric(),
                        Balanced_Accuracy = numeric(),
                        TP = integer(),
                        FP = integer(),
                        FN = integer(),
                        TN = integer())
  
  # Loop over each threshold and calculate the metrics
  for (threshold in threshold_options) {
    predicted_classes <- ifelse(scores > threshold, 1, -1)
    
    # Create confusion matrix, ensuring that both classes (-1 and 1) are represented
    confusion_matrix <- table(factor(predicted_classes, levels = c(-1, 1)),
                              factor(ground_truth_labels, levels = c(-1, 1)))
    
    # Extract values from confusion matrix or set to 0 if they don't exist
    TP <- ifelse("1" %in% rownames(confusion_matrix) && "1" %in% colnames(confusion_matrix),
                 confusion_matrix["1", "1"], 0)
    FP <- ifelse("1" %in% rownames(confusion_matrix) && "-1" %in% colnames(confusion_matrix),
                 confusion_matrix["1", "-1"], 0)
    FN <- ifelse("-1" %in% rownames(confusion_matrix) && "1" %in% colnames(confusion_matrix),
                 confusion_matrix["-1", "1"], 0)
    TN <- ifelse("-1" %in% rownames(confusion_matrix) && "-1" %in% colnames(confusion_matrix),
                 confusion_matrix["-1", "-1"], 0)
    
    # Calculate precision and recall
    precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
    recall <- ifelse(sum(ground_truth_labels == 1) == 0, 0, TP / sum(ground_truth_labels == 1))
    # Calculate recall for the negative class (-1)
    recall_neg <- ifelse(sum(ground_truth_labels == -1) == 0, 0, TN / sum(ground_truth_labels == -1))
    
    # Calculate accuracy
    accuracy <- (TP + TN) / sum(confusion_matrix)
    
    # Balanced Accuracy is the average of recall for positive and negative classes
    balanced_accuracy <- (recall + recall_neg) / 2
    
    # Calculate F1 score
    F1_score <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
    
    # Store the results
    results <- rbind(results, data.frame(threshold = threshold,
                                         F1_score = F1_score,
                                         Precision = precision,
                                         Recall = recall,
                                         Recall_neg = recall_neg,
                                         Accuracy = accuracy,
                                         Balanced_Accuracy = balanced_accuracy,
                                         TP = TP, FP = FP, FN = FN, TN = TN))
  }
  
  # Find the threshold that gives the best F1 score
  best_result <- results[which.max(results$F1_score), ]
  
  return(list(threshold = best_result$threshold,
              TP = best_result$TP,
              TN = best_result$TN,
              FP = best_result$FP,
              FN = best_result$FN,
              F1_score = best_result$F1_score,
              Precision = best_result$Precision,
              Recall = best_result$Recall,
              Accuracy = best_result$Accuracy,
              Balanced_Accuracy = best_result$Balanced_Accuracy))
}


