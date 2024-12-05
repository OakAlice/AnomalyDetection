# Testing highest performing hyperparmeters on the test set -----------------

# load in the best parameters (presuming you made them into a csv)
optimal_hyperparameters <- fread(file.path(base_path, "Output", "OptimalHyperparameters.csv"))
balance <- "stratified_balance"




# OCC models ----------------------------------------------------------------
OCC_hyperparameters <- optimal_hyperparameters %>% filter(data_name == dataset_name,
                                                          model_type == "OCC")
for (i in length(OCC_hyperparameters$activity)){

  # define the row you want to test
  parameter_row <- OCC_hyperparameters[i,]
  
  model_path <- file.path(base_path, "Output", "Models", paste0(parameter_row$data_name, "_", parameter_row$activity, "_final_model.rda"))
  
  if(file.exists(model_path)){
    print("model previously generated, loading in")
    load(model_path)
    top_features <- names(optimal_single_class_SVM$x.scale$`scaled:center`)
  } else {
    # Load in data
    training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv")))
    training_feature_data <- training_data %>% select(-c("OtherActivity", "GeneralisedActivity"))  
    
    # make a SVM with training data
    training_feature_data <- training_feature_data %>% mutate(Activity = ifelse(Activity == parameter_row$activity, Activity, "Other"))
    selected_feature_data <- featureSelection(training_feature_data, parameter_row$number_trees, parameter_row$number_features)
    target_selected_feature_data <- selected_feature_data[Activity == as.character(parameter_row$activity),!label_columns, with = FALSE]
    
  # create the optimal SVM 
    optimal_single_class_SVM <-
      do.call(
        svm,
        list(
          target_selected_feature_data,
          y = NULL,
          type = 'one-classification',
          nu = parameter_row$nu,
          scale = TRUE,
          kernel = parameter_row$kernel,
          gamma = parameter_row$gamma
        )
      )
    
    # save this model
   save(optimal_single_class_SVM, file = model_path)
   top_features <- colnames(target_selected_feature_data)
  }
  
  # load in the test data 
  testing_data <- fread(file = file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv")))
  
  if (file.exists(file.path(base_path, "Output", parameter_row$data_name, "_", parameter_row$activity, "_OCC_results.csv"))){
    print("results already calculated")
  } else{
    
    testing_feature_data <- testing_data %>% select(-c("OtherActivity", "GeneralisedActivity"))
    
    filtered_test_data <- testing_feature_data[, .SD, .SDcols = c("Activity", "Time", "ID", top_features)]
    filtered_test_data <- filtered_test_data[complete.cases(filtered_test_data),]
    
    # Balance the testing data
    if (balance == "non_stratified_balance"){
      activity_count <- filtered_test_data[Activity == target, .N]
      filtered_test_data[Activity  != parameter_row$activity,  Activity  := "Other"]
      filtered_test_data <- filtered_test_data[, .SD[1:activity_count], by = Activity]
    } else if (balance == "stratified_balance"){
      activity_count <- filtered_test_data[Activity == parameter_row$activity, .N] / length(unique(filtered_test_data$Activity))
      filtered_test_data <- filtered_test_data[, .SD[sample(.N, min(.N, activity_count))], by = Activity]
    }
    filtered_test_data <- filtered_test_data %>% arrange(ID, Time)
    
    # Extract ground truth labels
    ground_truth_labels <- ifelse(filtered_test_data$Activity == as.character(parameter_row$activity), 1, -1)
    numeric_features <- filtered_test_data[, .SD, .SDcols = !c("Activity", "Time", "ID")]
    
    # Predict using the trained model
    tryCatch({
      decision_scores <- predict(optimal_single_class_SVM, newdata = numeric_features, decision.values = TRUE)
      predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))
    }, error = function(e) {
      message <- paste("Error in predicting onto numeric data: ", e$message)
      model_features <- names(optimal_single_class_SVM$x.scale$`scaled:center`)
      data_features <- names(numeric_features)
      list(message = message, model_features = model_features, data_features = data_features, difference = difference)
    })
    
    # Calculate performance and save
    results <- calculatePerformance(predicted_scores, ground_truth_labels)
    results <- as.data.frame(results) %>% cbind("data_name" =parameter_row$data_name, "activity" =  parameter_row$activity)
    fwrite(results, file.path(base_path, "Output", paste0(parameter_row$data_name, "_", parameter_row$activity, "_OCC_results.csv")))
  }
  
  if (file.exists(file.path(base_path, "Output", "Predictions", paste0(parameter_row$data_name, "_", parameter_row$activity, "_OCC_predictions.csv")))){
    print("predictions already made")
  } else{
  
  # Classify predictions and save for plotting later
  full_test_data <- testing_data %>% select(all_of(c(top_features, "ID", "Activity", "Time"))) %>% na.omit()
  ground_truthed_labels <- full_test_data %>% select(c("Activity")) %>% mutate(Activity = ifelse(Activity == parameter_row$activity, 1, -1))
  numeric_features <- full_test_data %>% select(!c("Activity", "ID", "Time"))
  
  decision_scores <- predict(optimal_single_class_SVM, newdata = numeric_features, decision.values = TRUE)
  predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))
  
  predicted_classes <- ifelse(predicted_scores > results$threshold, 1, -1)
    
  # Save predictions with additional metadata
  outcome_classifications <- cbind(
    predicted_classes,
    ground_truthed_labels,
    full_test_data[, .SD, .SDcols = c("Activity", "ID", "Time")]
  )
  fwrite(outcome_classifications, file.path(base_path, "Output", "Predictions", paste0(parameter_row$data_name, "_", parameter_row$activity, "_OCC_predictions.csv")))
}

}




# Testing binary models ---------------------------------------------------

# Load in the best parameters (presuming you made them into a CSV)
Binary_hyperparameters <- fread(file.path(base_path, "Output", paste0(dataset_name, "_Binary_Hyperparameters.csv")))

# Loop through each activity in the parameter table
for (i in seq_len(nrow(Binary_hyperparameters))) {
  
  # Define the row you want to test
  parameter_row <- Binary_hyperparameters[i,]
  
  # Load in training data
  training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv")))
  training_data <- training_data %>%
    select(-c("OtherActivity", "GeneralisedActivity"))  # Remove unwanted columns
  
  # Update activity labels to binary (target activity vs. Other)
  training_data <- training_data %>%
    mutate(Activity = ifelse(Activity == parameter_row$activity, parameter_row$activity, "Other"))
  
  # Perform feature selection
  selected_feature_data <- featureSelection(training_data, parameter_row$number_trees, parameter_row$number_features)
  selected_feature_data <- na.omit(selected_feature_data)  # Remove rows with NA values
  
  
  fwrite(selected_feature_data, file.path(base_path, "Data", "Feature_data", "SpareFeatureData_Shaking_binary.csv"))
  
  # Create the optimal SVM model
  svm_args <- list(
    x = as.matrix(selected_feature_data[, setdiff(colnames(selected_feature_data), c("Activity", label_columns)), with = FALSE]),  # Features only
    y = as.factor(selected_feature_data$Activity),
    type = "C-classification",
    nu = parameter_row$nu,
    scale = TRUE,
    kernel = parameter_row$kernel,
    gamma = parameter_row$gamma
  )
  binary_SVM <- do.call(svm, svm_args)
  
  # Save the trained model
  model_path <- file.path(base_path, "Output", "Models", paste0(parameter_row$data_name, "_", parameter_row$activity, "_binary_final_model.rda"))
  save(binary_SVM, file = model_path)
  
  # Load in testing data
  testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv")))
  testing_feature_data <- testing_data %>%
    select(-c("OtherActivity", "GeneralisedActivity"))  # Remove unwanted columns
  
  # Filter test data to retain selected features
  top_features <- colnames(selected_feature_data)[!colnames(selected_feature_data) %in% c("Activity")]
  filtered_test_data <- testing_feature_data[, .SD, .SDcols = c("Activity", "Time", "ID", top_features)]
  filtered_test_data <- filtered_test_data[complete.cases(filtered_test_data),]
  
  # Balance the testing data if specified
  balance <- TRUE
  if (balance == TRUE) {
    activity_count <- filtered_test_data[Activity == parameter_row$activity, .N]
    filtered_test_data[Activity != parameter_row$activity, Activity := "Other"]
    filtered_test_data <- filtered_test_data[, .SD[1:activity_count], by = Activity]
  }
  
  # Extract ground truth labels
  ground_truth_labels <- ifelse(filtered_test_data$Activity == parameter_row$activity, 1, -1)
  numeric_features <- as.matrix(filtered_test_data[, !c("Activity", "Time", "ID"), with = FALSE])
  
  # Predict using the trained model
  decision_scores <- predict(binary_SVM, newdata = numeric_features, decision.values = TRUE)
  predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))
  
  # Calculate performance metrics
  results <- calculatePerformance(predicted_scores, ground_truth_labels)
  
  # Classify predictions
  predicted_classes <- ifelse(predicted_scores > results$threshold, 1, -1)
  
  # Save predictions with additional metadata
  outcome_classifications <- cbind(
    predicted_classes,
    ground_truth_labels,
    filtered_test_data[, .SD, .SDcols = c("Activity", "ID", "Time")]
  )
  fwrite(outcome_classifications, file.path(base_path, "Output", "Predictions", paste0(parameter_row$data_name, "_", parameter_row$activity, "_binary_predictions.csv")))
  
  # Print performance results for the current iteration for reviewing
  print(results)
}

# Multi-class models ------------------------------------------------------
Multi_hyperparameters <- fread(file.path(base_path, "Output", paste0(dataset_name, "_Multi_Hyperparameters.csv")))

for (i in length(Multi_hyperparameters$activity)){
  
  # i <- 1
  # define the row you want to test
  parameter_row <- Multi_hyperparameters[i,]
  
  training_data <-fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_multi_features.csv")))
  testing_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(parameter_row$data_name, "_test_multi_features.csv")))
  
  # select the right column for the testing activity based on multi, and remove the others
  training_feature_data <- update_feature_data(training_data, parameter_row$activity)
  training_feature_data <- training_feature_data[!Activity == ""]
  
  testing_feature_data <- update_feature_data(testing_data, parameter_row$activity)
  testing_feature_data <- testing_feature_data[!Activity == ""]
  
  # run this manually to get per class metrics
  baseline_results <- baselineMultiClass(dataset_name,
                                         condition = parameter_row$activity,
                                         training_data = training_feature_data,
                                         testing_data = testing_feature_data,
                                         number_trees = parameter_row$number_trees,
                                         number_features = parameter_row$number_features,
                                         kernel = parameter_row$kernel,
                                         gamma = parameter_row$gamma
                                         )
  
  print(baseline_results)
}

