
# Hyperparameter Optimisation ---------------------------------------------

# define the save paths
hyperparam_file <- file.path(base_path, "Output", "Tuning", paste0(dataset_name, "_OCCandBinary_hyperparmaters.csv"))
multi_hyperparam_file <- file.path(base_path, "Output", "Tuning", paste0(dataset_name, "_Multi_hyperparmaters.csv"))

# Define bounds for Bayesian Optimization of SVMs
bounds <- list(
  nu = c(0.001, 0.1),
  gamma = c(0.001, 0.1),
  kernel = c(1, 2, 3),
  number_trees = c(100, 500),
  number_features = c(10, 75)
)

# save the output 
save_best_params <- function(data_name, model_type, activity, elapsed_time, results) {
  data.frame(
    data_name = data_name,
    model_type = model_type,
    behaviour_or_activity = activity,
    elapsed = as.numeric(elapsed_time[3]),
    system = as.numeric(elapsed_time[2]),
    user = as.numeric(elapsed_time[1]),
    nu = results$Best_Par["nu"],
    gamma = results$Best_Par["gamma"],
    kernel = results$Best_Par["kernel"],
    number_trees = results$Best_Par["number_trees"],
    number_features = results$Best_Par["number_features"],
    Best_Value = results$Best_Value,
    Selected_Features = paste(
      unlist(results$Pred[[which(results$History$Value == results$Best_Value)]]), 
      collapse = ", ")
  )
}

save_results <- function(results_list, file_path) {
  results_df <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
  fwrite(results_df, file_path, row.names = FALSE)
}


model_type <- c("OCC", "Binary")

### TO DO: Add in binary logic for predictions ####

# Model tuning master call --------------------------------------------------
for (model in model_type){
  if(exists(file.path(base_path, "Output", "Tuning", paste0(dataset_name, "_", model, "_hyperparmaters.csv")))){
    message("Hyperparmeters already tuned for ", dataset_name," ", model, " models")
  } else {
  results_stored <- list()
  
  # load in the training data
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv"))) %>%
    select(-c("OtherActivity", "GeneralisedActivity")) %>%
    as.data.table()
  
  for (activity in target_activities) {
    print(paste("Tuning", model , "model for activity:", activity))
    
    plan(multisession, workers = availableCores() - 1)

    # Benchmark with a single iteration
    elapsed_time <- system.time({
        results <- BayesianOptimization(
          FUN = function(nu, gamma, kernel, number_trees, number_features) {
            modelTuning(
              model = model,
              activity = activity,
              feature_data = feature_data, 
              nu = nu,
              kernel = kernel,
              gamma = gamma,
              number_trees,
              number_features,
              validation_proportion = validation_proportion,
              balance = balance
            )
          },
          bounds = bounds,
          init_points = 5,
          n_iter = 10,
          acq = "ucb",
          kappa = 2.576 
        )
    })
    
    # clean up memory and return to normal processing
    gc()
    plan(sequential)
    
    # Save best parameters
    results_stored[[activity]] <- save_best_params(
      data_name = dataset_name, 
      model_type = "OCC", 
      activity = activity, 
      elapsed_time = elapsed_time, 
      results = results
    )
  }
  
  # Save the results and benchmark times and resources
  save_results(results_stored, file.path(base_path, "Output", "Tuning", paste0(dataset_name, "_", model, "_hyperparmaters.csv")))
  }
}

# Multiclass model tuning -------------------------------------------------
if (!file.exists(multi_hyperparam_file)) {
  print("Beginning optimization for multiclass models.")
  best_params_list <- list()
  
  behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv"))) %>%
    as.data.table()
  
  for (behaviours in behaviour_columns) {
    print(paste("Tuning multiclass model for behaviour column:", behaviours))
    
    # Prepare the multiclass data
    multiclass_data <- feature_data %>%
      select(-(setdiff(behaviour_columns, behaviours))) %>%
      rename("Activity" = !!sym(behaviours))
    
    if (behaviours == "GeneralisedActivity") {
      multiclass_data <- multiclass_data %>% filter(!Activity == "")
    }
    
    # Benchmark the time for each model's Bayesian optimization using bench::mark
    # run the search inside of a parallised loop
    plan(multisession, workers = detectCores() - 1)
    
    elapsed_time <- system.time({
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
    })
    
    gc()
    
    # Save best parameters for this behaviour
    results_stored[[activity]] <- save_best_params(
      data_name = dataset_name, 
      model_type = "Multi", 
      activity = behaviours, 
      elapsed_time = elapsed_time, 
      results = results
    )
  }
  
  plan(sequential)
  # Save the results and benchmarking times and resources
  save_results(best_params_list, multi_hyperparam_file)
} else {
  print("Multi-class parameters have already been tuned and saved")
}

