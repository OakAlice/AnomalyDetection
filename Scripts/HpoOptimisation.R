
# Hyperparameter Optimisation ---------------------------------------------

# define the save paths
occ_hyperparam_file <- file.path(base_path, "Output", paste0(dataset_name, "_OCC_hyperparmaters.csv"))
binary_hyperparam_file <- file.path(base_path, "Output", paste0(dataset_name, "_Binary_hyperparmaters.csv"))
multi_hyperparam_file <- file.path(base_path, "Output", paste0(dataset_name, "_Multi_hyperparmaters.csv"))

# Define bounds for Bayesian Optimization of SVMs
bounds <- list(
  nu = c(0.001, 0.1),
  gamma = c(0.001, 0.1),
  kernel = c(1, 2, 3),
  number_trees = c(100, 500),
  number_features = c(10, 75)
)

# save the output 
save_best_params <- function(dataset_name, model_type, activity_or_behaviour, results, bench_result) {
  data.frame(
    data_name = dataset_name,
    model_type = model_type,
    behaviour_or_activity = activity_or_behaviour,
    nu = results$Best_Par["nu"],
    gamma = results$Best_Par["gamma"],
    kernel = results$Best_Par["kernel"],
    number_trees = results$Best_Par["number_trees"],
    number_features = results$Best_Par["number_features"],
    Best_Value = results$Best_Value,
    Selected_Features <- results$Pred[which(results$History$Value == results$Best_Value), ],
    Time_minutes = as.numeric(bench_result$total_time),
    Memory_bytes = as.numeric(bench_result$mem_alloc),
    n_gc = as.numeric(bench_result$N_gc)
  )
}

save_results <- function(results_list, file_path) {
  results_df <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
  fwrite(results_df, file_path, row.names = FALSE)
}

# OCC model tuning --------------------------------------------------------
if (!file.exists(occ_hyperparam_file)) {
  results_stored <- list()
  
  # load in the training data
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv"))) %>%
    select(-c("OtherActivity", "GeneralisedActivity")) %>%
    as.data.table()
  
  for (activity in target_activities) {
    print(paste("Tuning OCC model for activity:", activity))
    
    plan(multisession, workers = availableCores() - 1)
    
    # Benchmark with a single iteration
    bench_result <- bench::mark(
      {
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
          init_points = 2,
          n_iter = 5,
          acq = "ucb",
          kappa = 2.576 
        )
        NULL  # Avoid unnecessary post-processing in bench::mark
      },
      iterations = 1,  # Run only once
      check = FALSE
    )
    
    # clean up memory
    gc()
    plan(sequential)
    
    # Save best parameters
    results_stored[[activity]] <- save_best_params(
      dataset_name, "OCC", activity, results, bench_result
    )
  }
  
  # Save the results and benchmark times and resources
  save_results(results_stored, occ_hyperparam_file)
} else {
  print("One-class parameters have already been tuned and saved")
}

# Binary model tuning ------------------------------------------------------
if (!file.exists(binary_hyperparam_file)) {
  best_params_list <- list()
  
  for (activity in target_activities) {
    print(paste("Tuning binary model for activity:", activity))
    
    # Benchmark with only a single iteration
    plan(multisession, workers = detectCores() - 1)
    
    bench_result <- bench::mark(
      {
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
        NULL  # Avoid unnecessary post-processing in bench::mark
      },
      iterations = 1,  # Run only once
      check = FALSE
    )
    gc()
    
    plan(sequential)
    
    # Save best parameters
    best_params_list[[activity]] <- save_best_params(
      dataset_name, "Binary", activity, bench_result
    )
  }
  
  # Save the results and benchmark times and resources
  save_results(best_params_list, binary_hyperparam_file)
} else {
  print("Binary parameters have already been tuned and saved")
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
    
    bench_result <- bench::mark(
      {
        BayesianOptimization(
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
          init_points = 3,
          n_iter = 7,
          acq = "ucb",
          kappa = 2.576
        )
        NULL  # Avoid unnecessary storage in bench::mark
      },
      iterations = 1,  # Run only once
      check = FALSE
    )
    
    gc()
    
    # Save best parameters for this behaviour
    best_params_list[[behaviours]] <- save_best_params(
      dataset_name, "Multi", behaviours, bench_result
    )
  }
  
  plan(sequential)
  # Save the results and benchmarking times and resources
  save_results(best_params_list, multi_hyperparam_file)
} else {
  print("Multi-class parameters have already been tuned and saved")
}

