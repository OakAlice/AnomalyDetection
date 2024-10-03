# ---------------------------------------------------------------------------
# Assorted functions
# ---------------------------------------------------------------------------

source(file.path("C:/Users/oaw001/Documents/AnomalyDetection", "Scripts", "UserInput.R"))

# ensure a directory exists
ensure.dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

# threshold metrics
threshold_metrics <- function(scores, ground_truth_labels, threshold) {
  # Assign predicted classes based on threshold
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
  
  # Calculate accuracy
  accuracy <- (TP + TN) / sum(confusion_matrix)
  
  # Calculate recall for the negative class (-1)
  recall_neg <- ifelse(sum(ground_truth_labels == -1) == 0, 0, TN / sum(ground_truth_labels == -1))
  
  # Balanced Accuracy is the average of recall for positive and negative classes
  balanced_accuracy <- (recall + recall_neg) / 2
  
  # Calculate F1 score
  F1_score <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  
  return(list(threshold = threshold,
              F1_score = F1_score,
              Precision = precision,
              Recall = recall,
              Accuracy = accuracy,
              Balanced_Accuracy = balanced_accuracy))
}

