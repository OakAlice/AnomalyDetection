# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data 3                ####
# ---------------------------------------------------------------------------
# Hyperparmeter Optimisation, one-class and multi-class

# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.001, 0.1),
  gamma = c(0.001, 0.1),
  kernel = c(1, 2, 3),
  number_trees = c(100, 500),
  number_features = c(10, 75)
)

# Tuning OCC model hyperparameters --------------------------------------
# PR-AUC for the target class is optimised
if(file.exists(file.path(base_path, "Output", paste0(dataset_name, "_OCC_Hyperparameters.csv")))){
  print("optimal hyperparameter doc has already been generated for OCC models")
} else {
  
  best_params_list <- list()
  
  for (activity in target_activities) {
    print(activity)
    
    # Load and preprocess feature data
    feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv")))
    feature_data <- feature_data %>% select(-c("OtherActivity", "GeneralisedActivity")) %>% as.data.table()
    
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = function(nu, gamma, kernel, number_trees, number_features) {
        OCCModelTuning(
          feature_data = feature_data,
          target_activity = activity, 
          nu = nu,
          kernel = kernel,
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
      },
      bounds = bounds,
      init_points = 5,
      n_iter = 10,
      acq = "ucb",
      kappa = 2.576 
    )
    
    # Extract the best parameter set and add to the list
    best_params <- data.frame(
      data_name <- dataset_name,
      model_type = "OCC",
      activity = activity,
      nu = results$Best_Par["nu"],
      gamma = results$Best_Par["gamma"],
      kernel = results$Best_Par["kernel"],
      number_trees = results$Best_Par["number_trees"],
      number_features = results$Best_Par["number_features"],
      value = results$Best_Value
    )
    
    best_params_list[[activity]] <- best_params
  }
  
  best_params_df <- do.call(rbind, best_params_list)
  
  fwrite(best_params_df, file.path(base_path, "Output", "OptimalOCCHyperparmeters.csv"), row.names = FALSE)
}


# Tuning binary models ----------------------------------------------------
# EWWWWW before I'm told I should do this, I will just do it
if(file.exists(file.path(base_path, "Output", paste0(dataset_name, "_Binary_Hyperparameters.csv")))){
  print("optimal hyperparameter doc has already been generated for binary models")
} else {
  
  best_params_list <- list()
  
  for (activity in target_activities) {
    print(activity)
    
    # Load and preprocess feature data
    feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv")))
    feature_data <- feature_data %>% select(-c("OtherActivity", "GeneralisedActivity")) %>% as.data.table()
    
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = function(nu, gamma, kernel, number_trees, number_features) {
        binaryModelTuning(
          feature_data = feature_data,
          target_activity = activity, 
          nu = nu,
          kernel = kernel,
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
      },
      bounds = bounds,
      init_points = 5,
      n_iter = 10,
      acq = "ucb",
      kappa = 2.576 
    )
    
    # Extract the best parameter set and add to the list
    best_params <- data.frame(
      data_name <- dataset_name,
      model_type = "OCC",
      activity = activity,
      nu = results$Best_Par["nu"],
      gamma = results$Best_Par["gamma"],
      kernel = results$Best_Par["kernel"],
      number_trees = results$Best_Par["number_trees"],
      number_features = results$Best_Par["number_features"],
      value = results$Best_Value
    )
    
    best_params_list[[activity]] <- best_params
  }
  
  best_params_df <- do.call(rbind, best_params_list)
  
  fwrite(best_params_df, file.path(base_path, "Output", "OptimalOCCHyperparmeters.csv"), row.names = FALSE)
}




# Tuning multiclass model hyperparameters ---------------------------------
# this section of the code tunes the multiclass model, optimising macro average F1
# the same bounds as in the above section are used for comparison sake
# remmeber to account for there being multiple types of activity columns 
if (file.exists(file.path(base_path, "Output", paste0(dataset_name, "_Multi_Hyperparameters.csv")))) {
  print("Optimal hyperparameter document has already been generated for Multi models")
} else {
  
  print("beginning optimisation of hyperparameters for multi models")
  
  behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv")))
  feature_data <- feature_data %>% as.data.table()
  
  best_params_list <- list()
  
    for (behaviours in behaviour_columns) {
      print(behaviours)
      
      multiclass_data <- feature_data %>%
        select(-(setdiff(behaviour_columns, behaviours))) %>%
        rename("Activity" = !!sym(behaviours))
      
      if (behaviours == "GeneralisedActivity") {
        multiclass_data <- multiclass_data %>% filter(!Activity == "")
      }
      
      # Run the Bayesian Optimization
      results <- BayesianOptimization(
        FUN = function(nu, gamma, kernel, number_trees, number_features) {
          multiclassModelTuning(
            multiclass_data = multiclass_data,
            nu = nu,
            kernel = kernel,
            gamma = gamma,
            number_trees = number_trees,
            number_features = number_features
          )
        },
        bounds = bounds,
        init_points = 5,
        n_iter = 10,
        acq = "ucb",
        kappa = 2.576 
      )
      
      # Extract the best parameters and add to the list
      best_params <- data.frame(
        data_name <- dataset_name,
        model_type = "Multi",
        behaviour = behaviours,
        nu = results$Best_Par["nu"],
        gamma = results$Best_Par["gamma"],
        kernel = results$Best_Par["kernel"],
        number_trees = results$Best_Par["number_trees"],
        number_features = results$Best_Par["number_features"],
        value = results$Best_Value
      )
      best_params_list[[behaviours]] <- best_params

  }
  
  best_params_df <- do.call(rbind, best_params_list)
  
  # Save the dataframe as a CSV file
  fwrite(best_params_df, file.path(base_path, "Output", paste0(dataset_name, "_Multi_Hyperparameters.csv")), row.names = FALSE)
}
