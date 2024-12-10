#  Functions for tuning the model hyperparameters across k_fold loops --------


adjust_activity <- function(data, model, activity) {
  # Ensure the input is a data.table
  data <- data.table::as.data.table(data)

  # Adjust the Activity column based on the model type
  data[, Activity := ifelse(Activity == activity, activity, "Other")]

  # For OCC model, retain only the target activity and "Other"
  if (model == "OCC") {
    data <- data[Activity %in% c(activity, "Other")]
  }

  return(data)
}

ensure_activity_representation <- function(validation_data, model, retries = 10) {
  retry_count <- 0
  while (sum(validation_data$Activity == activity) == 0 && retry_count < retries) {
    retry_count <- retry_count + 1
    message(activity, " not represented in validation fold. Retrying... (Attempt ", retry_count, ")")
    test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
    validation_data <- feature_data[ID %in% test_ids]
    validation_data <- adjust_activity(validation_data, model, activity)
  }
  if (retry_count == retries) stop("Unable to find a valid validation split after ", retries, " attempts.")
  return(validation_data)
}

balance_data <- function(data) {
  activity_count <- data[data$Activity == activity, .N] / length(unique(data$Activity))
  data[, .SD[sample(.N, min(.N, activity_count))], by = Activity]
}

split_data <- function(model, activity, balance, feature_data, validation_proportion) {
  # Ensure feature_data is a data.table
  setDT(feature_data)

  unique_ids <- unique(feature_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))

  training_data <- feature_data[!ID %in% test_ids]
  validation_data <- feature_data[ID %in% test_ids]

  # Balance validation and training data
  if (balance == "stratified_balance") {
    validation_data <- balance_data(validation_data)
    training_data <- balance_data(training_data)
  }

  # Adjust training and validation data based on the model type
  training_data <- adjust_activity(training_data, model, activity)
  validation_data <- adjust_activity(validation_data, model, activity)

  # Retry logic if the target activity is not represented
  validation_data <- ensure_activity_representation(validation_data, model)

  if (model == "OCC") {
    training_data <- training_data[training_data$Activity == activity, ]
  }

  return(list(training_data = training_data, validation_data = validation_data))
}



# Binary and 1-class model tuning function --------------------------------
modelTuning <- function(model, activity, feature_data, nu, kernel, gamma, number_features, validation_proportion, balance) {
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
            selected_training_data <- featureSelection(training_data, number_features, corr_threshold = 0.8)
            selected_training_data <- na.omit(selected_training_data)

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
            top_features <- colnames(selected_training_data)

            selected_validation_data <- validation_data[, .SD, .SDcols = top_features]
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

            message("precision: ", precision_metric)
            message("recall: ", recall_metric)

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

      message("about to stitch the loops together")

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


# Multiclass Model Tuning -------------------------------------------------
multiclassModelTuning <- function(multiclass_data, nu, kernel, gamma, number_trees, number_features, validation_proportion, loops) {
  # Convert kernel from numeric to kernel type
  kernel_type <- ifelse(kernel < 0.5, "linear",
    ifelse(kernel < 1.5, "radial", "polynomial")
  )

  # Parallelise loop over iterations
  num_loops <- 1:loops
  f1_scores <- future_lapply(num_loops, function(i) {
    tryCatch(
      {
        set.seed(i)
        # Split data into training and validation sets
        data_split <- split_data(multiclass_data, validation_proportion)
        training_data <- data_split$training_data
        validation_data <- data_split$validation_data

        # Feature selection
        selected_feature_data <- featureSelection(training_data, number_trees, number_features)
        selected_feature_data <- na.omit(selected_feature_data) # Clean the data

        # Train SVM model with class balancing
        class_weights <- table(selected_feature_data$Activity)
        class_weights <- max(class_weights) / (class_weights + 1e-6) # Prevent division by zero

        svm_args <- list(
          x = as.matrix(selected_feature_data[, !("Activity"), with = FALSE]),
          y = as.factor(selected_feature_data$Activity),
          type = "C-classification",
          nu = nu,
          scale = TRUE,
          kernel = kernel_type,
          gamma = gamma,
          class.weights = class_weights
        )
        multiclass_SVM <- do.call(svm, svm_args)

        # Validate the model
        top_features <- colnames(selected_feature_data)
        validation_features <- validation_data[, ..top_features]
        validation_features <- na.omit(validation_features) # Clean validation data
        numeric_validation_data <- as.matrix(validation_features[, !("Activity"), with = FALSE])
        ground_truth_labels <- validation_features$Activity

        # Predictions
        predictions <- predict(multiclass_SVM, newdata = numeric_validation_data)

        # Compute confusion matrix and F1-scores
        confusion_matrix <- table(predictions, ground_truth_labels)
        all_classes <- sort(union(rownames(confusion_matrix), colnames(confusion_matrix)))
        conf_matrix_padded <- matrix(0,
          nrow = length(all_classes), ncol = length(all_classes),
          dimnames = list(all_classes, all_classes)
        )
        conf_matrix_padded[rownames(confusion_matrix), colnames(confusion_matrix)] <- confusion_matrix

        # Calculate the F1 scores using confusionMatrix function
        metrics <- confusionMatrix(conf_matrix_padded)
        f1 <- metrics$byClass[, "F1"]
        f1[is.na(f1)] <- 0 # Replace NAs with 0
        return(mean(f1)) # Return the macro F1 score from this iteration
      },
      error = function(e) {
        message("Error during iteration ", i, ": ", e$message)
        return(NA) # Return NA when error
      }
    )
  }, future.seed = TRUE)

  # Return average F1-score
  average_macro_f1 <- mean(f1_scores, na.rm = TRUE)

  # Return results
  return(list(Score = average_macro_f1, Pred = top_features))
}
