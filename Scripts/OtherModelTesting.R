
# Other model -------------------------------------------------------------

# Multiclass model tuning -------------------------------------------------
# Define different behaviour column groupings
behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")

# Load feature data for multiclass optimization
feature_data <- fread(
  file.path(base_path, "Data", "Feature_data", 
            paste0(dataset_name, "_other_features.csv"))
) %>%
  as.data.table()

# Optimise for each behaviour grouping at a time
behaviours <- "OtherActivity"

    # Prepare data for current grouping
    multiclass_data <- feature_data %>%
      select(-(setdiff(behaviour_columns, behaviours))) %>%
      rename("Activity" = !!sym(behaviours))
    
    # Perform Bayesian optimization for multiclass model
    elapsed_time <- system.time({
      results <- BayesianOptimization(
        FUN = function(nu, gamma, kernel, number_features) {
          multiclassModelTuningUnweighted(
            model = "Multi",
            multiclass_data = multiclass_data,
            nu = nu,
            kernel = kernel,
            gamma = gamma,
            number_features = number_features,
            validation_proportion = validation_proportion,
            balance = balance,
            loops = 1 # only repeat once
          )
        },
        bounds = bounds,
        init_points = 5,
        n_iter = 10,
        acq = "ucb",
        kappa = 2.576 
      )
    })
    
    gc()
    # plan(sequential)
    
    # Store results for this grouping
    results_stored <- save_best_params(
      data_name = dataset_name,
      model_type = "Multi_Unweighted",
      activity = behaviours,
      elapsed_time = elapsed_time,
      results = results
    )
  
  # Save all multiclass results
  results_df <- rbind(results_stored)
  fwrite(results_df, file.path(base_path, "Output", "Tuning", 
                               paste0(dataset_name, "_Multi_", behaviours, "_unweighted_hyperparmaters.csv")),  row.names = FALSE)
  

multiclassModelTuningUnweighted <- function(model, multiclass_data, nu, kernel, gamma, 
                                  number_features, 
                                  validation_proportion, balance, loops) {
  tryCatch({
    model_outcomes <- list()
    
    # Convert kernel from numeric to kernel type
    kernel_type <- 
      ifelse(kernel < 0.5, "linear",
             ifelse(kernel < 1.5, "radial", "polynomial")
      )
    
    # Sequential version so I dont have the issue when parallelising
    future_outcomes <- lapply(num_loops, function(i) {
      tryCatch({
        set.seed(i)
        
        # Split data into training and validation sets
        data_split <- tryCatch({
          split_data(
            model = "Multi", 
            activity = "not_needed", 
            balance = balance, 
            feature_data = multiclass_data, 
            validation_proportion = validation_proportion
          )
        }, error = function(e) {
          message("Error in data splitting: ", e$message)
          return(NULL)
        })
        
        training_data <- data_split$training_data
        validation_data <- data_split$validation_data
        
        message("data split")
        flush.console()
        
        # Feature selection
        top_features <- tryCatch({
          featureSelection(
            model = "Multi", 
            training_data, 
            number_features, 
            corr_threshold = 0.8
          )
        }, error = function(e) {
          message("Error in feature selection: ", e$message)
          return(NULL)
        })
        
        
        selected_training_data <- tryCatch({
          training_data[, ..top_features]
        }, error = function(e) {
          message("Error selecting features from training data: ", e$message)
          return(NULL)
        })
        
        if (is.null(selected_training_data)) {
          stop("Failed to select features from training data")
        }
        
        selected_training_data <- na.omit(selected_training_data)
        
        message("features selected")
        flush.console()
        
        # SVM model training
        multiclass_SVM <- tryCatch({
          svm_args <- list(
            x = as.matrix(selected_training_data[, !("Activity"), with = FALSE]),
            y = as.factor(selected_training_data$Activity),
            type = "C-classification",
            nu = nu,
            scale = TRUE,
            kernel = kernel_type,
            gamma = gamma
          )
          do.call(svm, svm_args)
        }, error = function(e) {
          message("Error in SVM training: ", e$message)
          return(NULL)
        })
        
        message("model trained")
        flush.console()
        
        # Validation data preparation
        multiclass_test_data <- prepare_test_data(validation_data, selected_features = c(top_features, "Activity"), behaviour = NULL)
        numeric_validation_data <- as.data.frame(multiclass_test_data$numeric_data)
        ground_truth_labels <- multiclass_test_data$ground_truth_labels
        
        # Predictions and performance calculation
        predictions_and_metrics <- tryCatch({
          #numeric_validation_data <- numeric_validation_data[, !("Activity"), with = FALSE]
          
          # Handle invalid rows - important for seal data, don't remove
          invalid_row_indices <- which(!complete.cases(numeric_validation_data) |
                                         !apply(numeric_validation_data, 1, function(row) all(is.finite(row))))
          
          if (length(invalid_row_indices) > 0) {
            numeric_validation_data <- numeric_validation_data[-invalid_row_indices, , drop = FALSE]
            ground_truth_labels <- ground_truth_labels[-invalid_row_indices]
          }
          
          predictions <- predict(multiclass_SVM, newdata = numeric_validation_data)
          
          # find the per-class metrics and then average to macro
          macro_multiclass_scores <- multiclass_class_metrics(ground_truth_labels, predictions)
          macro_metrics <- macro_multiclass_scores$macro_metrics
          
          list(macro_f1 = macro_metrics$F1_Score, top_features = paste(top_features, collapse = ", "))
        }, error = function(e) {
          message("Error in predictions and metrics calculation: ", e$message)
          return(NULL)
        })
        
        if (is.null(predictions_and_metrics)) {
          stop("Failed to calculate predictions and metrics")
        }
        
        return(predictions_and_metrics)
        
      }, error = function(e) {
        message("Error during iteration ", i, ": ", e$message)
        return(NULL)
      })
    })
    
    # Process results
    valid_outcomes <- Filter(Negate(is.null), future_outcomes)
    
    if (length(valid_outcomes) == 0) {
      stop("No valid outcomes from any iteration")
    }
    
    model_outcomes <- tryCatch({
      rbindlist(valid_outcomes, fill = TRUE)
    }, error = function(e) {
      message("Error combining results: ", e$message)
      return(NULL)
    })
    
    if (is.null(model_outcomes)) {
      stop("Failed to combine model outcomes")
    }
    
    # Calculate final metrics
    avg_outcome <- tryCatch({
      mean_f1 <- mean(as.numeric(model_outcomes$macro_f1), na.rm = TRUE)
      selected_features <- model_outcomes$top_features[1]  # Take features from first valid run
      
      list(Score = mean_f1, Pred = selected_features)
    }, error = function(e) {
      message("Error calculating final metrics: ", e$message)
      return(list(Score = NA, Pred = NA))
    })
    
    return(avg_outcome)
    
  }, error = function(e) {
    message("Critical error in multiclassModelTuning: ", e$message)
    return(list(Score = NA, Pred = NA))
  })
}

