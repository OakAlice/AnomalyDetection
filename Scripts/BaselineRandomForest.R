#---------------------------------------------------------------------------
# Baseline performance of multi-class Random Forest (for comparison)   ####
#---------------------------------------------------------------------------

training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))


fifteen_class_feature_data <- training_data
three_class_feature_data <- training_data %>%
  mutate(Activity = ifelse(Activity %in% c('Walking', 'Eating'), Activity, 'Other')) %>%
  setDT()

k_folds <- 3
validation_proportion <- 0.2
number_trees <- 500
mtry <- 3


labelled_data <- fifteen_class_feature_data



  potential_features <- select_potential_features(labelled_data, threshold = 0.9)
  features_and_columsn <- c(potential_features, "Activity", "ID")
  labelled_data <- labelled_data[, ..features_and_columsn]
  labelled_data$Activity <- as.factor(labelled_data$Activity)
  labelled_data <- labelled_data[complete.cases(labelled_data), ]
  
  unique_ids <- unique(labelled_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
  
  validation_data <- labelled_data %>%
    filter(ID %in% test_ids) %>%
    select(-ID)
  
  training_data <- labelled_data %>%
    filter(!ID %in% test_ids) %>%
    select(-ID)
  
  # Train Random Forest model
  rf_model <- randomForest(Activity ~ ., 
                           data = training_data, 
                           ntree = number_trees, 
                           mtry = mtry, 
                           importance = TRUE)

  # validate
  predictions <- predict(rf_model, validation_data)
  confusion_matrix <- table(predictions, validation_data$Activity)

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

  