#' Isolation Forest Model Tuning for One-Class Classification
#'
#' This function performs model tuning for One-Class Classification (OCC) using Isolation Forest.
#' It includes feature selection, model training with isolation forest, and validation across multiple iterations.
#'
#' @param model Character. Should be "OCC" for this function
#' @param activity Character. The target activity/behavior to classify
#' @param feature_data Data frame. The input features and activity data
#' @param nodesize Integer. Minimum size of terminal nodes
#' @param variables Add descriptions here
#' @param validation_proportion Numeric. Proportion of data to use for validation (0-1)
#' @param balance Character. Type of class balancing to apply ("stratified_balance" or NULL)
#'
#' @return List with two elements:
#'   \item{Score}{Numeric. Average F1 score across iterations}
#'   \item{Pred}{Character vector. Selected feature names}

# Function to tune OCC models using Bayesian optimization
OCCModelTuningRF <- function(model, activity, feature_data, 
                             n_trees, max_depth,
                             validation_proportion, balance) {
  
  future_outcomes <- list()
  
  # Main processing block
  tryCatch({
    # Parallelize loop for 3 iterations 
    iterations <- 1:3
    future_outcomes <- future_lapply(iterations, function(i) {
      tryCatch({
        message(sprintf("\nStarting iteration %d", i))
        
        # Split data
        message("Splitting data")
        data_split <- split_data("OCC", activity, balance, feature_data, validation_proportion)
        if (is.null(data_split) || nrow(data_split$training_data) == 0) {
          stop("Data split returned NULL or empty training data")
        }
        
        training_data <- as.data.table(data_split$training_data)
        validation_data <- as.data.table(data_split$validation_data)
        
        if (balance == "stratified_balance") {
          training_data <- undersample(training_data, "Activity")
          validation_data <- undersample(validation_data, "Activity") 
        }
        
        message("Cleaning training data")
        
        # Feature selection
        top_features <- featureSelection(model = "OCC",
                                       training_data,
                                       number_features = NA,
                                       corr_threshold = 0.8,
                                       forest = FALSE)
        
        # Clean training data
        selected_training_data <- training_data[, .SD, .SDcols = c(top_features, "Activity")]
        selected_training_data <- clean_dataset(selected_training_data)
        
        # Clean validation data
        selected_validation_data <- validation_data[, .SD, .SDcols = c(top_features, "Activity")]
        selected_validation_data <- clean_dataset(selected_validation_data)
        
        message("Training model")
        flush.console()
        
        # Prepare training data
        target_training_data <- selected_training_data %>% filter(Activity == activity)
        train_num <- as.data.table(target_training_data)[, !("Activity"), with = FALSE]
        
        # Train isolation forest
        iso_forest <- isolation.forest(data = train_num,
                                      ntrees = n_trees,
                                      max_depth = max_depth,
                                      sample_size = 256,
                                      standardize_data = TRUE)
        
        message("Iso model trained successfully")
        
        # Prepare validation data and make predictions
        truth_labels <- selected_validation_data$Activity
        numeric_pred_data <- as.data.table(selected_validation_data)[, !("Activity"), with = FALSE]
        prediction_scores <- predict(iso_forest, numeric_pred_data)
        prediction_labels <- ifelse(prediction_scores > 0.5, "Other", activity)
        
        # Validate predictions
        if (length(prediction_labels) != length(truth_labels)) {
          stop(sprintf("Mismatch in lengths: predictions=%d, ground_truth=%d",
                      length(prediction_labels), length(truth_labels)))
        }
        
        # Create consistent factors
        unique_classes <- sort(unique(c(prediction_labels, truth_labels)))
        prediction_labels <- factor(prediction_labels, levels = unique_classes)
        ground_truth_labels <- factor(truth_labels, levels = unique_classes)
        
        # Calculate F1 score
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
          max_depth = max_depth,
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
    
    # Aggregate results
    model_outcomes <- unique(rbindlist(future_outcomes, use.names = TRUE, fill = TRUE))
    
    avg_outcomes <- model_outcomes %>%
      group_by(Activity, n_trees, max_depth) %>%
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
    gc()
    list(Score = NA, Pred = character(0))
  })
}

#' Decision Tree Model Tuning for Binary Classification
#'
#' This function performs model tuning for binary classification using Decision Trees.
#' It includes feature selection, model training with decision trees, and validation across multiple iterations.
#'
#' @param model Character. Should be "Binary" for this function
#' @param activity Character. The target activity/behavior to classify
#' @param feature_data Data frame. The input features and activity data
#' @param variables Integer. Add here
#' @param validation_proportion Numeric. Proportion of data to use for validation (0-1)
#' @param balance Character. Type of class balancing to apply ("stratified_balance" or NULL)
#'
#' @return List with two elements:
#'   \item{Score}{Numeric. Average F1 score across iterations}
#'   \item{Pred}{Character vector. Selected feature names}
#'
BinaryModelTuningRF <- function(model, activity, feature_data, 
                                min_samples_leaf, min_samples_split, max_depth,
                                validation_proportion, balance) {
  future_outcomes <- list()
  
  # Main processing block
  tryCatch({
    # Parallelize loop for 3 iterations
    iterations <- 1:3
    future_outcomes <- future_lapply(iterations, function(i) {
      tryCatch({
        message(sprintf("\nStarting iteration %d", i))
        
        # Data splitting and cleaning
        message("Splitting data")
        data_split <- split_data("Binary", activity, balance, feature_data, validation_proportion)
        if (is.null(data_split) || nrow(data_split$training_data) == 0) {
          stop("Data split returned NULL or empty training data")
        }
        
        # Convert and balance data if needed
        training_data <- as.data.table(data_split$training_data)
        validation_data <- as.data.table(data_split$validation_data)
        
        if (balance == "stratified_balance") {
          training_data <- undersample(training_data, "Activity")
          validation_data <- undersample(validation_data, "Activity")
        }
        
        # Feature selection and data preparation
        message("Cleaning training data")
        # remove the really bad columns # i.e., the fully NA and 0 ones.
        top_features <- featureSelection(
          model = "Binary",
          training_data,
          number_features = NA,
          corr_threshold = 0.8,
          forest = FALSE
        )
        
        # Clean and select features for both datasets
        selected_training_data <- training_data[, .SD, .SDcols = c(top_features, "Activity")] %>%
          clean_dataset()
        selected_validation_data <- validation_data[, .SD, .SDcols = c(top_features, "Activity")] %>%
          clean_dataset()
        
        # Model training
        message("Training model")
        flush.console()
        
        # stopped using this version as had less hyperparameters
        # trained_tree <- tree(
        #   formula = Activity ~ .,
        #   data = as.data.frame(selected_training_data),
        #   control = tree.control(
        #     nobs = nrow(selected_training_data),
        #     mindev = 0.01,
        #     minsize = min_samples_leaf, 
        #     mincut = min_samples_split
        #   )
        # )
        
        trained_tree <- rpart(
          formula = Activity ~ .,
          data = as.data.frame(selected_training_data),
          method = "class",  # Use "class" for classification
          control = rpart.control(
            maxdepth = max_depth,          # Maximum depth of the tree
            minsplit = min_samples_split,  # Minimum number of observations for a split
            minbucket = min_samples_leaf,  # Minimum number of observations in terminal nodes
            
            # Additional control parameters
            cp = 0.01,                     # Complexity parameter for pruning
            xval = 10                      # Number of cross-validations
          )
        )
        
        message("Binary model trained successfully")
        
        # Prediction and evaluation
        truth_labels <- selected_validation_data$Activity
        numeric_pred_data <- as.data.table(selected_validation_data)[, !("Activity"), with = FALSE]
        
        prediction_labels <- predict(trained_tree, newdata = as.data.frame(numeric_pred_data), type = "class")
        
        # Validation checks
        if (length(prediction_labels) != length(truth_labels)) {
          stop(sprintf("Mismatch in lengths: predictions=%d, ground_truth=%d", 
                       length(prediction_labels), length(truth_labels)))
        }
        
        # Create factors with consistent levels
        unique_classes <- sort(unique(c(prediction_labels, truth_labels)))
        prediction_labels <- factor(prediction_labels, levels = unique_classes)
        ground_truth_labels <- factor(truth_labels, levels = unique_classes)
        
        # table(prediction_labels, ground_truth_labels)
        
        f1_score <- MLmetrics::F1_Score(
          y_true = truth_labels,
          y_pred = prediction_labels,
          positive = activity
        )
        f1_score[is.na(f1_score)] <- 0
        
        # Return this iteration's results
        list(
          Activity = as.character(activity),
          min_samples_leaf = as.numeric(min_samples_leaf),
          min_samples_split = as.numeric(min_samples_split),
          max_depth = as.numeric(max_depth),
          f1 = as.numeric(f1_score),
          top_features = as.character(top_features)
        )
        
      }, error = function(e) {
        message(sprintf("Error in iteration %d: %s", i, e$message))
        NULL
      })
    }, future.seed = TRUE)
    
    # Process final results
    future_outcomes <- Filter(Negate(is.null), future_outcomes)
    
    if (length(future_outcomes) == 0) {
      message("No valid outcomes from iterations")
      return(list(Score = NA, Pred = character(0)))
    }
    
    # Convert to data frame without duplicates
    model_outcomes <- unique(rbindlist(future_outcomes, use.names = TRUE, fill = TRUE))
    avg_outcomes <- model_outcomes %>%
      group_by(Activity, min_samples_leaf, min_samples_split, max_depth) %>%
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

#' Random Forest Model Tuning for Multiclass Classification
#'
#' This function performs model tuning for multiclass classification using Random Forest.
#' It includes feature selection, model training, and validation across multiple iterations.
#'
#' @param model Character. Should be "Multiclass" for this function
#' @param multiclass_data Data frame. The input features and activity data
#' @param n_trees Integer. Number of trees in the random forest
#' @param mtry Integer. Number of variables randomly sampled at each split
#' @param min_node_size Integer. Minimum size of terminal nodes
#' @param sample_fraction Numeric. Fraction of observations to sample for each tree
#' @param validation_proportion Numeric. Proportion of data to use for validation (0-1)
#' @param balance Logical. Whether to balance the classes in training data
#' @param loops Integer. Number of iterations for cross-validation
#'
#' @return List with two elements:
#'   \item{Score}{Numeric. Average F1 score across iterations}
#'   \item{Pred}{Character vector. Selected feature names}
#'
multiclassModelTuningRF <- function(model, multiclass_data, 
                                   min_samples_split, min_samples_leaf, max_depth,
                                   validation_proportion, balance, loops) {
  
  future_outcomes <- list()
  
  # Main processing block
  tryCatch({
    message("Starting multiclass model tuning")
    
    # Split data
    message("Splitting data")
    data_split <- tryCatch({
      split_data("Multi", activity, balance, multiclass_data, validation_proportion)
    }, error = function(e) {
      message("Error in data splitting: ", e$message)
      return(NULL)
    })
    
    if (is.null(data_split) || nrow(data_split$training_data) == 0) {
      stop("Data split returned NULL or empty training data")
    }
    
    # Data preparation
    training_data <- as.data.table(data_split$training_data)
    validation_data <- as.data.table(data_split$validation_data)
    
    # Feature selection
    message("Performing feature selection")
    top_features <- tryCatch({
      featureSelection(model = "Multi",
                      training_data,
                      number_features = NA,
                      corr_threshold = 0.8,
                      forest = FALSE)
    }, error = function(e) {
      message("Error in feature selection: ", e$message)
      return(NULL)
    })
    
    if (is.null(top_features)) {
      stop("Feature selection failed")
    }
    
    # Clean and prepare datasets
    message("Cleaning and preparing datasets")
    tryCatch({
      selected_training_data <- training_data[, .SD, .SDcols = c(top_features, "Activity")]
      selected_training_data <- clean_dataset(selected_training_data)
      
      selected_validation_data <- validation_data[, .SD, .SDcols = c(top_features, "Activity")]
      selected_validation_data <- clean_dataset(selected_validation_data)
      
      train_num <- as.data.table(selected_training_data)[, !("Activity"), with = FALSE]
      ground_truth_labels <- as.factor(selected_training_data$Activity)
    }, error = function(e) {
      stop("Error in data preparation: ", e$message)
    })
    
    # Model training
    message("Training model")
    flush.console()
    
    tryCatch({
      class_weights <- table(selected_training_data$Activity)
      class_weights <- max(class_weights) / class_weights
      weights_vector <- class_weights[selected_training_data$Activity]
      
      # this is a decision tree version
      trained_tree <- rpart(
        formula = Activity ~ .,
        data = as.data.frame(selected_training_data),
        method = "class",  # Use "class" for classification
        weights = weights_vector,
        control = rpart.control(
          maxdepth = max_depth,          # Maximum depth of the tree
          minsplit = min_samples_split,  # Minimum number of observations for a split
          minbucket = min_samples_leaf,  # Minimum number of observations in terminal nodes
          cp = 0.01,                     # Complexity parameter for pruning
          xval = 10                      # Number of cross-validations
        )
      )
      
      # This is a random forest model # it performed too well
      # trained_tree <- tryCatch({
      #   class_weights <- table(selected_training_data$Activity)
      #   class_weights <- max(class_weights) / class_weights
      #   
      #   ranger(
      #     x = train_num, 
      #     y = ground_truth_labels,
      #     num.trees = n_trees,
      #     mtry = mtry, 
      #     min.node.size = min_samples_leaf, 
      #     max.depth = max_depth,
      #     class.weights = class_weights
      #   )
    
    }, error = function(e) {
      message("Error in model training: ", e$message)
      return(NULL)
    })
    
    if (is.null(trained_tree)) {
      stop("Model training failed")
    }
    
    message("Multi model trained successfully")
    
    # Predictions and evaluation
    tryCatch({
      truth_labels <- selected_validation_data$Activity
      numeric_pred_data <- as.data.table(selected_validation_data)[, !("Activity"), with = FALSE]
      
      # random forest
      # predictions <- predict(trained_tree, data = as.data.frame(numeric_pred_data))
      # prediction_labels <- predictions$predictions
      
      # decision tree version
      prediction_labels <- predict(trained_tree, newdata = as.data.frame(numeric_pred_data), type = "class")
      
      if (length(prediction_labels) != length(truth_labels)) {
        stop(sprintf("Mismatch in lengths: predictions=%d, ground_truth=%d", 
                    length(prediction_labels), length(truth_labels)))
      }
      
      # Create factors with consistent levels
      unique_classes <- sort(unique(c(prediction_labels, truth_labels)))
      prediction_labels <- factor(prediction_labels, levels = unique_classes)
      ground_truth_labels <- factor(truth_labels, levels = unique_classes)
      
      # table(ground_truth_labels, prediction_labels)
      
      # Calculate metrics
      macro_multiclass_scores <- multiclass_class_metrics(ground_truth_labels, prediction_labels)
      macro_metrics <- macro_multiclass_scores$weighted_metrics
      
      return(list(
        Score = macro_metrics$F1_Score, 
        Pred = paste(top_features, collapse = ", ")
      ))
      
    }, error = function(e) {
      message("Error in prediction and evaluation: ", e$message)
      return(list(Score = NA, Pred = character(0)))
    })
    
  }, error = function(e) {
    message("Critical error in main function: ", e$message)
    gc()
    return(list(Score = NA, Pred = character(0)))
  })
}
