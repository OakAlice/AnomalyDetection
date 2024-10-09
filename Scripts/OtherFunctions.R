# ---------------------------------------------------------------------------
# Assorted functions
# ---------------------------------------------------------------------------

# ensure a directory exists
ensure.dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
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



# check which packages are used from SibyllWang on Stack Exchange
checkPacks <- function(path) {
  
  # Get all R files in the directory
  # Note: Use the pattern ".R$" to match only files ending in .R
  files <- list.files(path)[str_detect(list.files(path), ".R$")]
  
  # Extract all functions and determine the package they come from using NCmisc::list.functions.in.file
  funs <- unlist(lapply(paste0(path, "/", files), list.functions.in.file))
  
  # Get the function names (which include package info)
  packs <- funs %>% names()
  
  # Identify "character" functions such as reactive objects in Shiny
  characters <- packs[str_detect(packs, "^character")]
  
  # Identify user-defined functions from the global environment
  globals <- packs[str_detect(packs, "^.GlobalEnv")]
  
  # Identify functions that are in multiple packages' namespaces
  multipackages <- packs[str_detect(packs, ", ")]
  
  # Extract unique package names from the multipackages
  mpackages <- multipackages %>%
    str_extract_all(., "[a-zA-Z0-9]+") %>%
    unlist() %>%
    unique()
  
  # Remove non-package elements from mpackages
  mpackages <- mpackages[!mpackages %in% c("c", "package")]
  
  # Identify functions that are from single packages
  packages <- packs[str_detect(packs, "package:") & !packs %in% multipackages] %>%
    str_replace(., "[0-9]+$", "") %>%
    str_replace(., "package:", "")
  
  # Get unique packages
  packages_u <- packages %>%
    unique() %>%
    union(., mpackages)
  
  # Return list of unique packages and their frequency table
  return(list(packs = packages_u, tb = table(packages)))
}
