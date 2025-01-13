#' Random Forest Model Tuning for Dichotomous Classification
#'
#' This function performs model tuning for either One-Class Classification (OCC) or Binary
#' Classification using Isolation Forest and Random Forest, respectively. It includes feature selection, model training, and
#' validation across multiple iterations.
#'
#' @param model Character. Either "OCC" or "Binary" specifying the classification type
#' @param activity Character. The target activity/behavior to classify
#' @param feature_data Data frame. The input features and activity data
#' @param n_trees Integer. Number of trees in the random forest
#' @param mtry Integer. Number of variables randomly sampled at each split
#' @param validation_proportion Numeric. Proportion of data to use for validation (0-1)
#' @param balance Logical. Whether to balance the classes in training data
#'
#' @return List with two elements:
#'   \item{Score}{Numeric. Average F1 score across iterations}
#'   \item{Pred}{Character vector. Selected feature names}

dichotomousModelTuningRF <- function(model, activity, feature_data, nodesize,
                                    validation_proportion, 
                                    balance) {
  # Input validation
  if (!model %in% c("OCC", "Binary")) {
    stop("Model must be either 'OCC' or 'Binary'")
  }
  
  # Setup parallel processing
  
  future_outcomes <- list()
  
  # Main processing block
  tryCatch({
    # Parallelize loop for 3 iterations
    iterations <- 1:3
    future_outcomes <- future_lapply(iterations, function(i) {
      tryCatch({
        # i <- 1
        
        # Set unique seed for each iteration
        message(sprintf("\nStarting iteration %d", i))
        
        # Data splitting
        message("Splitting data")
        data_split <- split_data(model, activity, balance, feature_data, validation_proportion)
        if (is.null(data_split) || nrow(data_split$training_data) == 0) {
          stop("Data split returned NULL or empty training data")
        }
        
        training_data <- as.data.table(data_split$training_data)
        validation_data <- as.data.table(data_split$validation_data)
        
        if (balance == "stratified_balance"){
          # balancing these between classes # since it's not working otherwise
          training_data <- undersample(training_data, "Activity")
          validation_data <- undersample(validation_data, "Activity")
        }
        
        message("Cleaning training data")
        
        # remove the really bad columns # i.e., the fully NA and 0 ones.
        top_features <- featureSelection(model = "Binary", 
                                         training_data, 
                                         number_features = NA, 
                                         corr_threshold = 0.8, 
                                         forest = FALSE)
        selected_training_data <- training_data[, .SD, .SDcols = c(top_features, "Activity")]
        selected_training_data <- selected_training_data[complete.cases(selected_training_data),]
        selected_training_data <- clean_dataset(selected_training_data)
        selected_training_data <- selected_training_data$data
        
        # Clean validation data to match
        selected_validation_data <- validation_data[, .SD, .SDcols = c(top_features, "Activity")]
        selected_validation_data <- clean_dataset(selected_validation_data)
        selected_validation_data <- selected_validation_data$data
        
        # Print summary of cleaning
        message(paste0("Training data rows: ", nrow(selected_training_data)))
        message(paste0("Validation data rows: ", nrow(selected_validation_data)))
        
        
        if (model == "Binary"){
          # use a binary decision tree
          # check that mtry isn't too big
          mtry <- ifelse(mtry > ncol(selected_training_data) - 1, ncol(selected_training_data) - 1, mtry)
  
          message("training model")
          flush.console()
          
          # Decision tree
          trained_tree <- tree(
            formula = Activity ~ .,
            data = as.data.frame(selected_training_data), # Ensure data is a data frame
            control = tree.control(
              nobs = nrow(selected_training_data), # Number of observations
              minsize = as.numeric(nodesize) * 2,  # Minimum size for splits
              mindev = 0.01                        # Minimum deviance for splits
            )
          )
        
          message("Model trained successfully")
          
          # select the important stuff
          truth_labels <- selected_validation_data$Activity
          numeric_pred_data <- as.data.table(selected_validation_data)[, !("Activity"), with = FALSE]
  
          # Make predictions
          prediction_labels <- predict(trained_tree, newdata = as.data.frame(numeric_pred_data), type = "class")
        } else if (model == "OCC"){
          
          target_training_data <- selected_training_data %>% filter(Activity == activity)
          train_num <- as.data.table(target_training_data)[, !("Activity"), with = FALSE]
          
          # Train Isolation Forest model
          iso_forest <- isolation.forest(train_num, 
                                         ntrees = n_trees, 
                                         sample_size = 256)
          
          # prep validation data
          truth_labels <- selected_validation_data$Activity
          numeric_pred_data <- as.data.table(selected_validation_data)[, !("Activity"), with = FALSE]
            
          prediction_scores <- predict(iso_forest, numeric_pred_data)
          
          # convert scores to classification (outliers vs. normal)
          prediction_labels <- ifelse(prediction_scores > 0.5, "Other", activity)
        }
        
        # Ensure we have matching valid data
        if (length(prediction_labels) != length(truth_labels)) {
          stop(sprintf("Mismatch in lengths: predictions=%d, ground_truth=%d", 
                      length(prediction_labels), length(truth_labels)))
        }
        
        # Create factors with consistent levels
        unique_classes <- sort(unique(c(prediction_labels, truth_labels)))
        prediction_labels <- factor(prediction_labels, levels = unique_classes)
        ground_truth_labels <- factor(truth_labels, levels = unique_classes)
        
        # Compute metrics
        f1_score <- MLmetrics::F1_Score(
          y_true = truth_labels, 
          y_pred = prediction_labels, 
          positive = activity
        )
        f1_score[is.na(f1_score)] <- 0
        
        # Return results
        list(
          Activity = as.character(activity),
          n_trees = as.numeric(n_trees),
          mtry = as.numeric(mtry),
          nodesize = as.numeric(nodesize),
          f1 = as.numeric(f1_score),
          top_features = as.character(top_features)
        )
        
      }, error = function(e) {
        message(sprintf("Error in iteration %d: %s", i, e$message))
        NULL
      })
    }, future.seed = TRUE)
    
    # Process results
    future_outcomes <- Filter(Negate(is.null), future_outcomes)
    
    if (length(future_outcomes) == 0) {
      message("No valid outcomes from iterations")
      return(list(Score = NA, Pred = character(0)))
    }
    
    # Convert to data frame without duplicates
    model_outcomes <- unique(rbindlist(future_outcomes, use.names = TRUE, fill = TRUE))
    
    avg_outcomes <- model_outcomes %>%
      group_by(Activity, n_trees, mtry, nodesize) %>%
      summarise(
        mean_F1 = mean(f1),
        .groups = 'drop'
      )
    
    list(
      Score = as.numeric(avg_outcomes$mean_F1)[1],
      Pred = if(length(future_outcomes) > 0) future_outcomes[[1]]$top_features else character(0)
    )
    
  }, error = function(e) {
    message("Critical error in main function: ", e$message)
    future::plan(future::sequential)
    gc()
    list(Score = NA, Pred = character(0))
  })
}






# for the multiclass scenario
# class_weights <- table(selected_training_data$Activity)
# class_weights <- max(class_weights) / class_weights
# 
# RF_args <- list(
#   x = numeric_training_data,
#   y = as.factor(selected_training_data$Activity),
#   n_trees = n_trees,
#   mtry = mtry,
#   min_node_size = min_node_size,
#   sample.fraction = sample_fraction,
#   scale = TRUE,
#   class.weights = class_weights
# )
# 
# trained_RF <- do.call(ranger, RF_args)





# trained_RF <- randomForest(
#   formula = Activity ~ .,
#   data = as.data.frame(selected_training_data),
#   ntree = as.numeric(n_trees), 
#   mtry = as.numeric(mtry),
#   nodesize = as.numeric(nodesize),
#   classwt = class_weights
# )