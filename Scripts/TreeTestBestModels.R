# Testing the best models -------------------------------------------------
# load in the test data
test_feature_data <- fread(file = file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

# make a minor formatting change
if (dataset_name == "Vehkaoja_Dog"){
  test_feature_data$GeneralisedActivity <- str_to_title(test_feature_data$GeneralisedActivity)
}

# Dichotomous models -----------------------------------------------------
for(model in c("OCC", "Binary")){
  test_results <- data.frame()
  for(behaviour in target_activities){
    # behaviour <- target_activities[2]
    
    # load in the model, comes in as "trained_SVM" or "trained_tree" - need to standardise
    load(file.path(base_path, "Output", "Models", ML_method, paste0(dataset_name, "_", model, "_", behaviour, "_final_model.rda")))
    
    trained_model <- trained_tree
    
    # extract the variables that were used in model development
    if (model == "OCC"){
      selected_features <- trained_model[["metadata"]][["cols_num"]]
    } else if (model == "Binary"){
      feature_string <- trained_model[["terms"]][[3]]
      selected_features <- str_split(as.character(feature_string), "\\+") %>%
        unlist() %>%
        str_trim()
      selected_features <- selected_features[selected_features != ""]
    }
    
    # prepare the test data
    # Select features and metadata columns
    selected_data <- test_feature_data[, .SD, .SDcols = c(selected_features, "Activity", "Time", "ID")]
    selected_data <- na.omit(as.data.table(selected_data))
    
    # Convert to binary classification
    selected_data$Activity <- ifelse(selected_data$Activity == behaviour, behaviour, "Other")
    
    # Extract components
    ground_truth_labels <- selected_data$Activity
    time_values <- selected_data$Time
    ID_values <- selected_data$ID
    numeric_data <- selected_data[, !c("Activity", "Time", "ID"), with = FALSE] %>% 
      mutate(across(everything(), as.numeric))
    
    # Remove invalid rows
    invalid_rows <- which(!complete.cases(numeric_data) |
                            !apply(numeric_data, 1, function(row) all(is.finite(row))))
    
    if (length(invalid_rows) > 0) {
      numeric_data <- numeric_data[-invalid_rows, , drop = FALSE]
      ground_truth_labels <- ground_truth_labels[-invalid_rows]
      time_values <- time_values[-invalid_rows]
      ID_values <- ID_values[-invalid_rows]
    }
    
    message("testing data prepared")
    
   if (model == "OCC"){
      decision_prob <- predict(trained_model, numeric_data)
      prediction_labels <- ifelse(decision_prob > 0.5, "Other", behaviour)
    } else if (model == "Binary"){
      prediction_labels <- predict(trained_model, newdata = as.data.frame(numeric_data), type = "class")
      probabilities <- predict(trained_model, newdata = as.data.frame(numeric_data), type = "prob")
      decision_prob <- probabilities[, behaviour]
    }
    
    message("predictions made")
    
    # Ensure predictions and ground_truth are factors with the same levels
    unique_classes <- sort(union(prediction_labels, ground_truth_labels))
    prediction_labels <- factor(prediction_labels, levels = unique_classes)
    ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)
    
    if (length(prediction_labels) != length(ground_truth_labels)) {
      stop("Error: Predictions and ground truth labels have different lengths.")
    }
    
    table(ground_truth_labels, prediction_labels)
    
    # Compute performance metrics
    f1_score <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = prediction_labels, positive = behaviour)
    precision_metric <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = prediction_labels, positive = behaviour)
    recall_metric <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = prediction_labels, positive = behaviour)
    accuracy_metric <- MLmetrics::Accuracy(y_true = ground_truth_labels, y_pred = prediction_labels)
    
    # Compute baseline metrics
    zero_rate_baseline <- as.list(calculate_zero_rate_baseline(ground_truth_labels, model, behaviour))
    random_baseline <- as.list(random_baseline_metrics(ground_truth_labels, iterations = 100, model))
    
    # Compile results for this run
    results <- data.frame(
      Dataset = as.character(dataset_name),
      Model = as.character(model),
      Activity = as.character(behaviour),
      Prevelance = NA, # just a space filler so it fits with the others
      F1_Score = as.numeric(f1_score),
      Precision = as.numeric(precision_metric),
      Recall = as.numeric(recall_metric),
      Accuracy = as.numeric(accuracy_metric),
      ZeroR_F1_Score = zero_rate_baseline$F1_Score,
      ZeroR_Precision = zero_rate_baseline$Precision,
      ZeroR_Recall = zero_rate_baseline$Recall,
      ZeroR_Accuracy = zero_rate_baseline$Accuracy,
      Random_F1_Score_prev = random_baseline$macro_summary$F1_Score_prev,
      Random_Precision_prev = random_baseline$macro_summary$Precision_prev,
      Random_Recall_prev = random_baseline$macro_summary$Recall_prev,
      Random_Accuracy_prev = random_baseline$macro_summary$Accuracy_prev,
      Random_F1_Score_equal = random_baseline$macro_summary$F1_Score_equal,
      Random_Precision_equal = random_baseline$macro_summary$Precision_equal,
      Random_Recall_equal = random_baseline$macro_summary$Recall_equal,
      Random_Accuracy_equal = random_baseline$macro_summary$Accuracy_equal
    )
    
    test_results <- rbind(test_results, results)
    message("test results stored")
    
    # write out the predictions for later plotting
    output <- data.table(
      "Time" = time_values,
      "ID" = ID_values,
      "Ground_truth" = ground_truth_labels, 
      "Predictions" = prediction_labels,
      "decision_prob" = as.vector(decision_prob)
    )
    fwrite(output, file.path(base_path, "Output", "Testing", ML_method, "Predictions", paste(dataset_name, model, behaviour, "predictions.csv", sep = "_")))
    message("predictions saved")
    
  }
  # save the results
  fwrite(test_results, file.path(base_path, "Output", "Testing", ML_method, paste0(dataset_name, "_", model, "_test_performance.csv")))
}


# Multi-class models ------------------------------------------------------

for(behaviour_set in c("Activity", "OtherActivity", "GeneralisedActivity")){
  
  # load in the model, comes in as "trained_SVM"
  load(file.path(base_path, "Output", "Models", ML_method, paste0(dataset_name, "_Multi_", behaviour_set, "_final_model.rda")))
  
  # extract the right variables from the model
  selected_features <- trained_tree$forest$independent.variable.names
  
  # prepare the test data
  multiclass_test_data <- update_feature_data(test_feature_data, behaviour_set)
  if (behaviour_set == "GeneralisedActivity") {
    multiclass_test_data <- multiclass_test_data %>% filter(!Activity == "")
  }
  
  multiclass_test_data <- prepare_test_data(multiclass_test_data, selected_features, behaviour = NULL)
  numeric_test_data <- as.data.frame(multiclass_test_data$numeric_data)
  ground_truth_labels <- multiclass_test_data$ground_truth_labels
  time_values <- multiclass_test_data$time_values
  ID_values <- multiclass_test_data$ID_values
  
  message("testing data prepared")
  
  # make predictions
  predictions <- predict(trained_tree, data = numeric_test_data)
  prediction_labels <- predictions$predictions
  message("predictions made")
  
  if (length(prediction_labels) != length(ground_truth_labels)) {
    stop("Error: Predictions and ground truth labels have different lengths.")
  }
  
  table(ground_truth_labels, prediction_labels)
  
  test_results <- calculate_full_multi_performance(ground_truth_labels, predictions = prediction_labels, model = behaviour_set)
  
  fwrite(test_results, file.path(base_path, "Output", "Testing", ML_method, paste0(dataset_name, "_Multi_", behaviour_set, "_test_performance.csv")), row.names = FALSE)
  message("test results stored")
  
  # Save predictions
  output <- data.table(
    "Time" = time_values,
    "ID" = ID_values,
    "Ground_truth" = ground_truth_labels, 
    "Predictions" = prediction_labels
  )
  
  if (nrow(output) > 0) {
    fwrite(output, file.path(base_path, "Output", "Testing", ML_method, "Predictions", paste(dataset_name, behaviour_set, "predictions.csv", sep = "_")))
    message("predictions saved")
  } else {
    message("No predictions to save.")
  }
}



# Read all files together -------------------------------------------------
test_files <- list.files(file.path(base_path, "Output", "Testing", ML_method), pattern = paste0(dataset_name, "_.*\\.csv$"), full.names = TRUE)
test_outcome <- rbindlist(
  lapply(test_files, function(file) {
    df <- fread(file)
    return(df)
  }),
  use.names = TRUE, fill=TRUE
)

test_outcome[is.na(test_outcome)] <- 0

test_outcome$Activity <- str_to_title(test_outcome$Activity) # format for consistency
# calculate the adjusted values
combined_results_adjusted <- test_outcome %>%
  mutate(Zero_adj_F1_Score = F1_Score - ZeroR_F1_Score,
         Zero_adj_Precision = Precision - ZeroR_Precision,
         Zero_adj_Recall = Recall - ZeroR_Recall,
         Zero_adj_Accuracy = Accuracy - ZeroR_Accuracy,
         Rand_adj_F1_Score_prev = F1_Score - Random_F1_Score_prev,
         Rand_adj_Precision_prev = Precision - Random_Precision_prev,
         Rand_adj_Recall_prev = Recall - Random_Recall_prev,
         Rand_adj_Accuracy_prev = Accuracy - Random_Accuracy_prev,
         Rand_adj_F1_Score_equal = F1_Score - Random_F1_Score_equal,
         Rand_adj_Precision_equal = Precision - Random_Precision_equal,
         Rand_adj_Recall_equal = Recall - Random_Recall_equal,
         Rand_adj_Accuracy_equal = Accuracy - Random_Accuracy_equal)

fwrite(combined_results_adjusted, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", ML_method, "_complete_test_performance.csv")))






