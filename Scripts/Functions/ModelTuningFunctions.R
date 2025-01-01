#| Model Tuning Functions

#' Tune Binary and One-Class Classification models
#' 
#' This function performs model tuning using cross-validation and returns
#' performance metrics for the given hyperparameter configuration.
#' 
#' @param model Model type ("OCC" or "Binary")
#' @param activity Target activity to classify
#' @param feature_data Input feature data
#' @param nu SVM nu parameter
#' @param kernel Kernel type (linear, radial, polynomial)
#' @param gamma Kernel coefficient
#' @param number_features Number of features to select
#' @param validation_proportion Proportion for validation set
#' @param balance Data balancing strategy
#' @return List containing Score (F1) and selected features
modelTuning <- function(model, activity, feature_data, 
                        nu, kernel, gamma, number_features, 
                        validation_proportion, balance) {
  tryCatch(
    {
      # Adjust kernel value
      kernel <- ifelse(kernel < 0.5, "linear", ifelse(kernel < 1.5, "radial", "polynomial"))

      # Parallelize loop for 3 iterations
      iterations <- 1:3
      future_outcomes <- future_lapply(iterations, function(i) {
        tryCatch(
          {
            set.seed(i)
            message(i)
            flush.console()

            # Split data into training and validation sets and formats them appropriate for OCC and binary respectively
            data_split <- split_data(model, activity, balance, feature_data, validation_proportion)
            training_data <- as.data.table(data_split$training_data)
            validation_data <- as.data.table(data_split$validation_data)

            message("split data")
            flush.console()

            # Feature selection
            top_features <- featureSelection(model, training_data, number_features, corr_threshold = 0.8)
            selected_training_data <- training_data[, .SD, .SDcols = c(top_features, "Activity")]
            selected_training_data <- selected_training_data[complete.cases(selected_training_data),]

            message("features selected")
            flush.console()

            # Train SVM model
            numeric_training_data <- selected_training_data[, !"Activity", with = FALSE]
            if (anyNA(numeric_training_data) || any(!is.finite(as.matrix(numeric_training_data)))) {
              stop("Training data contains invalid values (NA, NaN, or Inf).")
            }

            svm_args <- list(
              x = numeric_training_data,
              type = ifelse(model == "OCC", "one-classification", "C-classification"),
              nu = nu,
              scale = TRUE,
              kernel = kernel,
              gamma = gamma
            )
            # Add some extra things only when it's not OCC
            if (model != "OCC") {
              svm_args$y <- as.factor(selected_training_data$Activity)

              class_weights <- table(selected_training_data$Activity)
              class_weights <- max(class_weights) / class_weights

              svm_args$class.weights <- class_weights
            }

            trained_SVM <- do.call(svm, svm_args)

            message("trained models")
            flush.console()

            # Select features from the validation data
            selected_validation_data <- validation_data[, .SD, .SDcols = c(top_features, "Activity")]
            selected_validation_data <- na.omit(selected_validation_data)
            selected_validation_data <- as.data.table(selected_validation_data)

            # Generate predictions and calculate performance
            ground_truth_labels <- selected_validation_data$Activity
            numeric_validation_data <- selected_validation_data[, !("Activity"), with = FALSE]

            # this bit was important for the seal data, don't remove
            invalid_row_indices <- which(!complete.cases(numeric_validation_data) |
              !apply(numeric_validation_data, 1, function(row) all(is.finite(row))))

            if (length(invalid_row_indices) > 0) {
              numeric_validation_data <- numeric_validation_data[-invalid_row_indices, , drop = FALSE]
              ground_truth_labels <- ground_truth_labels[-invalid_row_indices]
            }

            predictions <- predict(trained_SVM, newdata = numeric_validation_data)

            message("predictions")

            if (model == "OCC") {
              predictions <- ifelse(predictions == FALSE, "Other", activity)
            }

            # Ensure predictions and ground_truth are factors with the same levels
            unique_classes <- sort(union(predictions, ground_truth_labels))
            predictions <- factor(predictions, levels = unique_classes)
            ground_truth_labels <- factor(ground_truth_labels, levels = unique_classes)

            if (length(predictions) != length(ground_truth_labels)) {
              stop("Error: Predictions and ground truth labels have different lengths.")
            }

            # Compute confusion matrix
            f1_score <- MLmetrics::F1_Score(y_true = ground_truth_labels, y_pred = predictions, positive = activity)
            precision_metric <- MLmetrics::Precision(y_true = ground_truth_labels, y_pred = predictions, positive = activity)
            recall_metric <- MLmetrics::Recall(y_true = ground_truth_labels, y_pred = predictions, positive = activity)

            # Replace NAs with 0
            f1_score[is.na(f1_score)] <- 0

            # Compile results for this run
            list(
              Activity = as.character(activity),
              nu = as.character(nu),
              gamma = as.character(gamma),
              kernel = as.character(kernel),
              number_features = ifelse(exists("number_features"), as.numeric(number_features), NA),
              F1_Score = as.numeric(f1_score),
              top_features = as.character(top_features)
            )
          },
          error = function(e) {
            message(sprintf("Error during run %d: %s", i, e$message))
            NULL
          }
        )
      }, future.seed = TRUE)

      # Filter out NULL results
      future_outcomes <- Filter(Negate(is.null), future_outcomes)

      # Check if any valid results exist
      if (length(future_outcomes) == 0) {
        message("No valid outcomes from the runs")
        return(list(Score = NA, Pred = NA))
      }

      model_outcomes <- rbindlist(future_outcomes, use.names = TRUE, fill = TRUE)

      # Calculate the average F1 score per configuration
      avg_outcomes <- model_outcomes[, .(mean_F1 = mean(F1_Score, na.rm = TRUE)),
        by = .(Activity, nu, gamma, kernel, number_features)
      ]

      # find all of the unique features that appeared in those 3 runs
      # selecting just one set might be overfit.
      top_features <- unique(model_outcomes$top_features)

      # Return the average F1 score and the top features
      list(Score = as.numeric(avg_outcomes$mean_F1), Pred = top_features)
    },
    error = function(e) {
      message("Error during model tuning: ", e$message)
      list(Score = NA, Pred = NA)
    }
  )
}

#' Tune Multi-class Classification models
#' 
#' Performs tuning for multi-class SVM models using cross-validation
#' and feature selection.
#' 
#' @param multiclass_data Input feature data
#' @param nu SVM nu parameter
#' @param kernel Kernel type
#' @param gamma Kernel coefficient
#' @param number_trees Number of trees for feature selection
#' @param number_features Number of features to select
#' @param validation_proportion Proportion for validation set
#' @param loops Number of cross-validation iterations
#' @return List containing Score (macro F1) and selected features
multiclassModelTuning <- function(model, multiclass_data, nu, kernel, gamma, 
                                number_features, 
                                validation_proportion, balance, loops) {
  tryCatch({
    model_outcomes <- list()

    # Convert kernel from numeric to kernel type
    kernel_type <- 
      ifelse(kernel < 0.5, "linear",
        ifelse(kernel < 1.5, "radial", "polynomial")
      )

    # Parallelise loop over iterations
    num_loops <- 1:loops
    # Parallel version (commented out)
    # future_outcomes <- future_lapply(num_loops, function(i) {

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

        if (is.null(data_split)) {
          stop("Data splitting failed")
        }

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

        if (is.null(top_features)) {
          stop("Feature selection failed")
        }

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

        # Train SVM model with class balancing
        class_weights <- tryCatch({
          weights <- table(selected_training_data$Activity)
          max(weights) / (weights + 1e-6)  # Prevent division by zero
        }, error = function(e) {
          message("Error calculating class weights: ", e$message)
          return(NULL)
        })

        if (is.null(class_weights)) {
          stop("Failed to calculate class weights")
        }

        # SVM model training
        multiclass_SVM <- tryCatch({
          svm_args <- list(
            x = as.matrix(selected_training_data[, !("Activity"), with = FALSE]),
            y = as.factor(selected_training_data$Activity),
            type = "C-classification",
            nu = nu,
            scale = TRUE,
            kernel = kernel_type,
            gamma = gamma,
            class.weights = class_weights
          )
          do.call(svm, svm_args)
        }, error = function(e) {
          message("Error in SVM training: ", e$message)
          return(NULL)
        })

        if (is.null(multiclass_SVM)) {
          stop("SVM model training failed")
        }

        message("model trained")
        flush.console()

        # Validation data preparation
        validation_features <- tryCatch({
          val_features <- validation_data[, ..top_features]
          val_features <- na.omit(val_features)
          as.data.table(val_features)
        }, error = function(e) {
          message("Error preparing validation data: ", e$message)
          return(NULL)
        })

        if (is.null(validation_features)) {
          stop("Validation data preparation failed")
        }

        # Predictions and performance calculation
        predictions_and_metrics <- tryCatch({
          ground_truth_labels <- validation_features$Activity
          numeric_validation_data <- validation_features[, !("Activity"), with = FALSE]

          # Handle invalid rows - important for seal data, don't remove
          invalid_row_indices <- which(!complete.cases(numeric_validation_data) |
            !apply(numeric_validation_data, 1, function(row) all(is.finite(row))))

          if (length(invalid_row_indices) > 0) {
            numeric_validation_data <- numeric_validation_data[-invalid_row_indices, , drop = FALSE]
            ground_truth_labels <- ground_truth_labels[-invalid_row_indices]
          }

          predictions <- predict(multiclass_SVM, newdata = numeric_validation_data)

          # Compute confusion matrix and metrics
          confusion_matrix <- table(predictions, ground_truth_labels)
          all_classes <- sort(union(rownames(confusion_matrix), colnames(confusion_matrix)))
          conf_matrix_padded <- matrix(0,
            nrow = length(all_classes), ncol = length(all_classes),
            dimnames = list(all_classes, all_classes)
          )
          conf_matrix_padded[rownames(confusion_matrix), colnames(confusion_matrix)] <- confusion_matrix

          metrics <- confusionMatrix(conf_matrix_padded)
          f1 <- metrics$byClass[, "F1"]
          f1[is.na(f1)] <- 0

          list(macro_f1 = mean(f1), top_features = paste(top_features, collapse = ", "))
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
