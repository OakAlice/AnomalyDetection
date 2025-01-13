# Training final models ------------------------------------------------

# SVM  -----------------------------------------------------------------

training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv"))) %>%
  as.data.table()

if (ML_method == "SVM"){
    
  for (model in c("OCC", "Binary", "Multi")) {
    
    # model <- "OCC"
    
    # Load hyperparameter file
    hyperparam_file <- fread(file = file.path(base_path, "Output", "Tuning", ML_method, paste0(dataset_name, "_", model, "_hyperparmaters.csv")))
  
  
    # hyperparam_file <- fread(file = file.path(base_path, "Output", "Tuning", "previous", paste0(dataset_name, "_", model, "_hyperparmaters.csv")))
    
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
      } else { # adjust for multi # they will all be named "Activity" - need to sub for the column we really want
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
      selected_training_data_y <- as.factor(selected_training_data$Activity)
  
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
        svm_args$y <- selected_training_data_y
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
}


# Random Forest -----------------------------------------------------------
training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv"))) %>%
  as.data.table()

if (ML_method == "Tree"){
  
  for (model in c("OCC", "Binary")) { #, "Multi")) {
    
    # model <- "Binary"
    
    # Load hyperparameter file
    hyperparam_file <- fread(file = file.path(base_path, "Output", "Tuning", ML_method, paste0(dataset_name, "_", model, "_hyperparmaters.csv")))
    
    for (i in seq_len(nrow(hyperparam_file))) {
      parameter_row <- hyperparam_file[i, ]
      
      activity <- parameter_row$behaviour_or_activity
      
      # Extract selected features
      selected_features <- strsplit(parameter_row$Selected_Features, ",\\s*")[[1]] %>% unique()
      selected_training_data <- training_data[, ..selected_features] # activity is hidden in here
      
      # Adjust the labels for OCC or Binary models
      if (parameter_row$model_type == "OCC"){
        
        target_training_data <- as.data.table(selected_training_data)
        selected_training_data[, Activity := ifelse(Activity == activity, activity, "Other")]
        target_training_data <- target_training_data[Activity == activity]
        
      } else if (parameter_row$model_type == "Binary"){
        
        target_training_data <- as.data.table(selected_training_data)
        selected_training_data[, Activity := ifelse(Activity == activity, activity, "Other")]
        target_training_data <- target_training_data[Activity %in% c(activity, "Other")]
        
      } else if (parameter_row$model_type == "Multi") { # adjust for multi # they will all be named "Activity" - need to sub for the column we really want
        
        selected_training_data <- selected_training_data %>% select(-Activity)
        behaviour_set <- parameter_row$behaviour_or_activity
        selected_training_data <- cbind(selected_training_data, "Activity" = training_data[[behaviour_set]])
        
        if (parameter_row$behaviour_or_activity == "GeneralisedActivity") {
          selected_training_data <- selected_training_data %>% filter(!Activity == "")
        }
      }
      
      # Prepare input features (ensure numeric and remove Activity column)
      selected_training_data_x <- selected_training_data %>%
        select(-Activity) %>%
        mutate(across(everything(), as.numeric))
      Activity <- selected_training_data$Activity
      
      # Remove invalid rows
      invalid_rows <- which(!complete.cases(selected_training_data_x) |
                              !apply(selected_training_data_x, 1, function(row) all(is.finite(row))))
      
      if (length(invalid_rows) > 0) {
        selected_training_data_x <- selected_training_data_x[-invalid_rows, , drop = FALSE]
        Activity <- Activity[-invalid_rows]
        time_values <- time_values[-invalid_rows]
        ID_values <- ID_values[-invalid_rows]
        selected_training_data <- cbind(selected_training_data_x, Activity)
      }
      
      message("Training data prepared.")
      
      if (parameter_row$model_type == "OCC"){
        
        trained_tree <- isolation.forest(selected_training_data_x, 
                                       ntrees = parameter_row$n_trees, 
                                       sample_size = 256)
        
      } else if (parameter_row$model_type == "Binary"){
        
        trained_tree <- tree(
          formula = Activity ~ .,
          data = as.data.frame(selected_training_data),
          control = tree.control(
            nobs = nrow(selected_training_data),
            minsize = as.numeric(parameter_row$nodesize) * 2, 
            mindev = 0.01                        
          )
        )
        
      } else if (parameter_row$model_type == "Multi"){
        
        print("haven't added this yet")
        
      }
      
      # Save the trained model
      model_path <- file.path(
        base_path, "Output", "Models", ML_method,
        paste0(parameter_row$data_name, "_", model, "_", parameter_row$behaviour_or_activity, "_final_model.rda")
      )
      save(trained_tree, file = model_path)
      message("Model saved")
      
      gc()
    }
  }
}





