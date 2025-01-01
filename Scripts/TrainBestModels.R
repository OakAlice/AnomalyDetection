# Testing highest performing hyperparmeters on the test set -----------------

# Training the final models -----------------------------------------------

training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv"))) %>%
  as.data.table()

for (model in c("OCC", "Binary", "Multi")) {
  # Load hyperparameter file
  hyperparam_file <- fread(file = file.path(base_path, "Output", "Tuning", paste0(dataset_name, "_", model, "_hyperparmaters.csv")))

  for (i in seq_len(nrow(hyperparam_file))) {
    parameter_row <- hyperparam_file[i, ]

    # Extract selected features
    selected_features <- strsplit(parameter_row$Selected_Features, ",\\s*")[[1]] %>% unique()
    selected_training_data <- training_data[, ..selected_features] # activity is hidden in here

    # Adjust the labels for OCC or Binary models
    if (!parameter_row$model_type == "Multi"){
      selected_training_data <- adjust_activity(
        data = selected_training_data,
        model = parameter_row$model_type,
        activity = parameter_row$behaviour_or_activity
      )
    } else { # adjust for multi
      selected_training_data <- selected_training_data %>% select(-Activity)
      behaviour_set <- parameter_row$behaviour_or_activity
      selected_training_data <- cbind(selected_training_data, "Activity" = training_data[[behaviour_set]])
      
      if (parameter_row$behaviour_or_activity == "GeneralisedActivity") {
        selected_training_data <- selected_training_data %>% filter(!Activity == "")
      }
    }

    # Filter for OCC model
    if (parameter_row$model_type == "OCC") {
      selected_training_data <- selected_training_data[Activity == parameter_row$behaviour_or_activity]
    }

    # Remove rows with missing values
    selected_training_data <- na.omit(selected_training_data)

    # Prepare input features (ensure numeric and remove Activity column)
    selected_training_data_x <- selected_training_data %>%
      select(-Activity) %>%
      mutate(across(everything(), as.numeric))

    message("Training data prepared.")

    # Define SVM arguments
    svm_args <- list(
      x = selected_training_data_x,
      type = ifelse(parameter_row$model_type == "OCC", "one-classification", "C-classification"),
      nu = parameter_row$nu,
      scale = TRUE,
      kernel = case_when(
        parameter_row$kernel < 0.5 ~ "linear",
        parameter_row$kernel < 1.5 ~ "radial",
        TRUE ~ "polynomial"
      ),
      gamma = parameter_row$gamma
    )

    # Add response variable and class weights for Binary and Multiclass model
    if (!parameter_row$model_type == "OCC") {
      svm_args$y <- as.factor(selected_training_data$Activity)
      class_weights <- table(selected_training_data$Activity)
      svm_args$class.weights <- max(class_weights) / class_weights
    }

    # Train the SVM model
    trained_SVM <- do.call(svm, svm_args)
    message("Model trained.")
    
    # Save the trained model
    model_path <- file.path(
      base_path, "Output", "Models",
      paste0(parameter_row$data_name, "_", model, "_", parameter_row$behaviour_or_activity, "_final_model.rda")
    )
    save(trained_SVM, file = model_path)
    message("Model saved.")
    
    gc()
  }
}
