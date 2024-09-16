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
  # firstly eliminate features that don't contribute anything (no variance, high correlation)
  potential_features <- select_potential_features(training_data, threshold = 0.9)
  features_and_columsn <- c(potential_features, "Activity", "Time", "ID")
  training_data <- training_data[, ..features_and_columsn]
  
  label_columns <- c("Activity", "Time", "ID")
  
  # now using the more fancy methods
  if (feature_selection == "UMAP") {
    training_data <- training_data[complete.cases(training_data), ]
    training_labels <- training_data[, Activity]
    training_numeric <- training_data[, .SD, .SDcols = setdiff(names(training_data), label_columns)]
    
    # not working, 16/09/2024
    UMAP_representations <- UMAP_reduction(numeric_features = training_numeric,
                                           labels = training_labels,
                                           minimum_distance = options_row$min_dist,
                                           num_neighbours = options_row$n_neighbours,
                                           shape_metric = options_row$metric,
                                           save_model_path = file.path(base_path, "Output", dataset_name))
    
    selected_feature_data <- as.data.frame(UMAP_representations$UMAP_2D_embeddings)
    
  } else if (feature_selection == "RF") {
    RF_features <- RF_feature_selection(training_data, target_column = "Activity", 
                                        n_trees = options_row$number_trees, 
                                        number_features = options_row$number_features)
    
    top_features <- RF_features$Selected_Features$Feature[1:number_features]
    all_columns <- c(top_features, label_columns)
    selected_feature_data <- training_data[, ..all_columns]
  }
  
  #### Train model ####
  if (options_row$model == "SVM") {
    params <- list(gamma = options_row$gamma, degree = options_row$degree) %>% compact()
    target_class_feature_data <- selected_feature_data[Activity == as.character(options_row$targetActivity)[1],
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
    selected_validation_data <- validation_data[, ..top_features]
    selected_validation_data <- selected_validation_data[complete.cases(selected_validation_data), ]
  }
  
  
  
  ### HERE ###
  
  ground_truth_labels <- validation_data %>% na.omit() %>% select(Activity)
  ground_truth_labels <- ifelse(ground_truth_labels == as.character(options_row$targetActivity), 1, -1)
  
  decision_scores <- predict(single_class_SVM, newdata = selected_validation_data, decision.values = TRUE)
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
  
  # do this 1:k_fold times
  cross_outcome <- map_dfr(1:k_folds, ~perform_single_validation(.x, subset_data, 
                                                                                  validation_proportion, 
                                                                                  feature_selection, 
                                                                                  options_row, 
                                                                                  base_path, 
                                                                                  dataset_name, 
                                                                                  number_features))
  
  # average these cross-validation results
  cross_outcome <- as.data.table(cross_outcome)
  cross_average <- cross_outcome[, .(
    mean_AUC = mean(AUC_Value, na.rm = TRUE),
    sd_AUC = sd(AUC_Value, na.rm = TRUE)
  ), by = setdiff(names(cross_outcome), "AUC_Value")]
  
  return(cross_average)
}
