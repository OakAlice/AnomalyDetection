# Training final models ---------------------------------------------------
# Random Forest -----------------------------------------------------------
training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", training_set, "_other_features.csv"))) %>%
  as.data.table()

for (model in c("OCC", "Binary", "Multi")) {
    
    # model <- "Multi"
    
    # Load hyperparameter file
    hyperparam_file <- fread(file = file.path(base_path, "Output", "Tuning", ML_method, paste0(dataset_name, "_", training_set, "_", model, "_hyperparmaters.csv")))
    
    for (i in seq_len(nrow(hyperparam_file))) {
      parameter_row <- hyperparam_file[i, ]
      
      activity <- parameter_row$behaviour_or_activity
      
      # Extract selected features
      selected_features <- strsplit(parameter_row$Selected_Features, ",\\s*")[[1]] %>% unique()
      selected_training_data <- training_data[, ..selected_features] # activity is hidden in here
      
      # Adjust the labels for OCC or Binary models
      if (parameter_row$model_type == "OCC"){
        
        target_training_data <- as.data.table(selected_training_data)
        target_training_data <- target_training_data[Activity == activity]
        
      } else if (parameter_row$model_type == "Binary"){
        
        target_training_data <- as.data.table(selected_training_data)
        target_training_data[, Activity := ifelse(Activity == activity, activity, "Other")]
        
      } else if (parameter_row$model_type == "Multi") { # adjust for multi # they will all be named "Activity" - need to sub for the column we really want
        
        target_training_data <- selected_training_data %>% select(-Activity)
        behaviour_set <- parameter_row$behaviour_or_activity
        target_training_data <- cbind(target_training_data, "Activity" = training_data[[behaviour_set]])
        
        if (parameter_row$behaviour_or_activity == "GeneralisedActivity") {
          target_training_data <- target_training_data %>% filter(!Activity == "")
        }
      }
      
      # Prepare input features (ensure numeric and remove Activity column)
      numeric_training_data <- target_training_data %>%
        select(-Activity) %>%
        mutate(across(everything(), as.numeric))
      Activity <- target_training_data$Activity
      
      # Remove invalid rows
      invalid_rows <- which(!complete.cases(numeric_training_data) |
                              !apply(numeric_training_data, 1, function(row) all(is.finite(row))))
      
      if (length(invalid_rows) > 0) {
        numeric_training_data <- numeric_training_data[-invalid_rows, , drop = FALSE]
        Activity <- Activity[-invalid_rows]
      }
      
      # recombine post mutations and cleaning
      target_training_data <- cbind(numeric_training_data, Activity)
      target_training_data$Activity <- as.factor(target_training_data$Activity)
      
      message("Training data prepared. Beginning model training.")
      
      if (parameter_row$model_type == "OCC"){
        
        n_trees <- abs(round(as.numeric(parameter_row$n_trees), 0))
        trained_tree <- isolation.forest(numeric_training_data, 
                                         ntrees = n_trees, 
                                         sample_size = 256)
        
      } else if (parameter_row$model_type == "Binary"){
        
        if (balance == "stratified_balance"){
          # balancing these between classes # since it's not working otherwise
          target_training_data <- undersample(target_training_data, "Activity")
        }
        
        trained_tree <- rpart(
          formula = Activity ~ .,
          data = as.data.frame(target_training_data),
          method = "class",  # Use "class" for classification
          control = rpart.control(
            maxdepth = parameter_row$max_depth,
            minsplit = parameter_row$min_samples_split,
            minbucket = parameter_row$min_samples_leaf, 
            
            cp = 0.01, 
            xval = 10 
          )
        )
        
      } else if (parameter_row$model_type == "Multi"){
        
        class_weights <- table(target_training_data$Activity)
        class_weights <- max(class_weights) / class_weights
        weights_vector <- class_weights[target_training_data$Activity]
        
        trained_tree <- tryCatch({
         rpart(
            formula = Activity ~ .,
            data = as.data.frame(target_training_data),
            method = "class",  # Use "class" for classification
            weights = weights_vector,
            control = rpart.control(
              maxdepth = max_depth,       
              minsplit = min_samples_split,
              minbucket = min_samples_leaf,
              cp = 0.01, 
              xval = 10
            )
          )
          
        }, error = function(e) {
          message("Error in model training: ", e$message)
          return(NULL)
        })
      }
      
      # Save the trained model
      model_path <- file.path(
        base_path, "Output", "Models", ML_method,
        paste0(parameter_row$data_name, "_", training_set, "_", model, "_", parameter_row$behaviour_or_activity, "_final_model.rda")
      )
      save(trained_tree, file = model_path)
      message("Model saved")
      
      gc()
  }
}
