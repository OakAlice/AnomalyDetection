#---------------------------------------------------------------------------
# Calculate performance under different conditions                      ####
#---------------------------------------------------------------------------

finalModelPerformance <- function(mode, training_data, optimal_model, testing_data = NULL, target_activity = NULL) {
  # Prepare data based on mode (training, testing, or randomized)
  if (mode == "training") {
    # For training, all labels are from the taregt class
    ground_truth_labels <- rep(1, nrow(training_data)) # All are positive class
    # the target_selected_feature_data has already been formatted above so don't need to do anything
    numeric_features <- training_data 
  } else {
    # For testing or randomization, seperate the numeric from the Activity
    top_features <- colnames(training_data)
    
    filtered_test_data <- testing_data[, .SD, .SDcols = c("Activity", top_features)]
    filtered_test_data <- filtered_test_data[complete.cases(filtered_test_data),]
    
    # if randomised, randomise the order of the Activity column
    if (mode == "random") {
      filtered_test_data$Activity <- sample(filtered_test_data$Activity)
    }
    
    # Balance the testing data
    activity_count <- filtered_test_data[Activity  == target_activity, .N]
    filtered_test_data[Activity  != target_activity,  Activity  := "Other"]
    balanced_test_data <- filtered_test_data[, .SD[1:activity_count], by = Activity]
  
    # Extract ground truth labels
    ground_truth_labels <- ifelse(balanced_test_data$Activity == as.character(target_activity), 1, -1)
    numeric_features <- balanced_test_data[, !"Activity"]
  }
  
  # Predict using the trained model
  decision_scores <- predict(optimal_model, newdata = numeric_features, decision.values = TRUE)
  predicted_scores <- as.numeric(attr(decision_scores, "decision.values"))
  
  # Calculate performance
  results <- calculatePerformance(predicted_scores, ground_truth_labels)
  
  return(results)
}
