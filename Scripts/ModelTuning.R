# ---------------------------------------------------------------------------
# Functions for tuning the model hyperparameters across k_fold validations
# ---------------------------------------------------------------------------

# train and validate these specific values 
perform_single_validation <- function(k, subset_data, validation_proportion, 
                                      feature_selection, options_row, base_path, 
                                      dataset_name, number_features) {
  
  # Create training and validation data
  unique_ids <- unique(subset_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
  validation_data <- subset_data[ID %in% test_ids]
  training_data <- subset_data[!ID %in% test_ids]
  
  #### Feature selection ####
  selected_feature_data <- feature_selection(training_data, options_row)
  
  #### Train model ####
  if (options_row$model == "SVM") {
    params <- list(gamma = options_row$gamma, degree = options_row$degree.x) %>% compact()
    params <- Filter(Negate(is.na), params)
    target_class_feature_data <- selected_feature_data[Activity == as.character(options_row$targetActivity),
                                                       !label_columns, with = FALSE] 

    single_class_SVM <- do.call(svm, c(list(target_class_feature_data, y = NULL, type = 'one-classification', 
                                            nu = options_row$nu, scale = TRUE, kernel = options_row$kernel), params))
  }
  
  #### Validate model ####
  if (feature_selection == "UMAP") {
    validation_numeric <- validation_data %>% select(-Activity, -Time, -ID)
    umap_model <- readRDS(file.path(base_path, "Output", "umap_2D_model.rds"))
    selected_validation_data <- predict(umap_model, validation_numeric) %>% as.data.frame()
    colnames(selected_validation_data) <- c("UMAP1", "UMAP2")
    
  } else if (feature_selection == "RF") {
    selected_validation_data <- validation_data[, .SD, .SDcols = c("Activity", top_features)]
    selected_validation_data <- selected_validation_data[complete.cases(selected_validation_data), ]
  }
  
  ground_truth_labels <- selected_validation_data[, "Activity"]
  ground_truth_labels <- ifelse(ground_truth_labels == as.character(options_row$targetActivity), 1, -1)
  
  numeric_validation_data <- selected_validation_data[, !"Activity"]
  decision_scores <- predict(single_class_SVM, newdata = numeric_validation_data, decision.values = TRUE)
  scores <- as.numeric(attr(decision_scores, "decision.values"))
  
  # Calculate AUC
  roc_curve <- roc(as.vector(ground_truth_labels), scores)
  auc_value <- auc(roc_curve)
  
  # Create a tibble for the cross-validation result
  cross_result <- tibble(
    Model = as.character(options_row$model),
    Activity = as.character(options_row$targetActivity),
    nu = as.character(options_row$nu),
    gamma = as.character(options_row$gamma),
    kernel = as.character(options_row$kernel),
    min_dist = as.character(options_row$min_dist),
    n_neighbours = as.character(options_row$n_neighbours),
    metric = as.character(options_row$metric),
    number_features = as.character(options_row$number_features),
    number_trees = as.character(options_row$number_trees),
    feature_method = as.character(feature_selection),
    AUC_Value = auc_value
  )
  
  return(cross_result)
}


# for each row in the options_df, average the AUC results of the k_folds
process_row <- function(options_row, k_folds, subset_data, validation_proportion, feature_selection, base_path, dataset_name, number_features) {
  
  # do this 1:k_fold times in parallel
  plan(multisession, workers = parallel::detectCores())
  
  # Perform parallel cross-validation, k-fold times. Can be parallel as each split is independent
  cross_outcome <- future_lapply(1:k_folds, function(k) {
    perform_single_validation(k, 
                              subset_data, 
                              validation_proportion, 
                              feature_selection, 
                              options_row, 
                              base_path, 
                              dataset_name, 
                              number_features)
  }, future.seed = TRUE)  # Set future.seed = TRUE within the call so the numbers are safe
  
  # Combine the results
  cross_results <- rbindlist(cross_outcome)
  
  # make sequential again
  plan(sequential)
  
  # average these cross-validation results
  suppressMessages(
    cross_average <- cross_results %>%
      # Exclude 'AUC_Value' from the grouping columns
      group_by(across(-AUC_Value)) %>%  
      mutate(AUC_Value = as.numeric(AUC_Value)) %>%
      summarise(
        mean_AUC = mean(AUC_Value, na.rm = TRUE),  # Calculate the mean AUC
        sd_AUC = sd(AUC_Value, na.rm = TRUE)       # Calculate the standard deviation of AUC
      ))
  
  return(cross_average)
}
