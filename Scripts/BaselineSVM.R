#---------------------------------------------------------------------------
# Baseline performance of multi-class Random Forest (for comparison)   ####
#---------------------------------------------------------------------------

training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

number_trees <- 500
mtry <- 3


  potential_features <- select_potential_features(training_data, threshold = 0.9)
  features_and_columsn <- c(potential_features, "Activity", "ID")
  training_data <- training_data[, ..features_and_columsn]
  training_data$Activity <- as.factor(training_data$Activity)
  training_data <- training_data[complete.cases(training_data), ]
 
  # Tune multiclass SVM parameters ### TODO
  
  # Train multiclass SVM with chosen parameters 
  kernel <- "radial"
  cost <- 0.092827
  gamma <- 0.08586
  
  svm_model <- svm(Activity ~ ., 
                   data = training_data, 
                   type = 'C-classification',   
                   kernel = kernel,           
                   cost = cost,           
                   gamma = gamma)
  
  
  # validate
  testing_data <- testing_data[, ..features_and_columsn]
  testing_data <- testing_data[complete.cases(testing_data), ]
  numeric_testing_data <- testing_data %>% select(!"Activity")
  
  decision_scores <- predict(svm_model, newdata = numeric_testing_data, decision.values = TRUE)
  scores <- as.numeric(attr(decision_scores, "decision.values"))
  ground_truth_labels <- testing_data$Activity
  
  training_results <- get_performance(scores, ground_truth_labels)
  
  
  
  
  
  
  
  
  
  confusion_matrix <- table(predictions, testing_data$Activity)

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

  