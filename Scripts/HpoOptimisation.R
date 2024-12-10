#| Hyperparameter Optimisation Script
#|
#| This script performs Bayesian optimization to find optimal hyperparameters for:
#| One-Class Classification (OCC) models and Binary Classification models
#| as welulti-class Classification models
#| Saves optimization results to CSV files in the Output/Tuning directory.

# Define parameter bounds for Bayesian Optimization
# nu: Controls training error/support vector ratio (0.01 to 0.5)
# gamma: RBF kernel parameter (0.0001 to 1)
# kernel: Kernel type (1=linear, 2=radial, 3=polynomial)
# number_features: Number of features to select (5 to 100)
bounds <- list(
  nu = c(0.01, 0.5),
  gamma = c(0.0001, 1),
  kernel = c(1, 2, 3),
  number_features = c(5, 100)
)

# Dichotomous Model tuning master call --------------------------------------
# Can currently do binary and OCC, will expand to multi-class as well

model_types <- c("OCC", "Binary")

for (model in model_types) {
  hyperparam_path <- file.path(
    base_path, "Output", "Tuning",
    paste0(dataset_name, "_", model, "_hyperparmaters.csv")
  )
  if (exists(hyperparam_path)) {
    message("Hyperparmeters already tuned for ", dataset_name, " ", model, " models")
  } else {
    results_stored <- list()

    # Load and prepare feature data
    feature_data <- fread(
      file.path(base_path, "Data", "Feature_data", 
                paste0(dataset_name, "_multi_features.csv"))
    ) %>%
      select(-c("OtherActivity", "GeneralisedActivity")) %>%
      as.data.table()

    # Optimize hyperparameters for each activity
    for (activity in target_activities) {
      print(paste("Tuning", model, "model for activity:", activity))

      # Set up parallel processing
      plan(multisession, workers = availableCores() - 1)

      # Perform Bayesian optimization
      # - init_points: Initial random points to evaluate
      # - n_iter: Number of optimization iterations
      # - acq: Acquisition function
      # - kappa: Trade-off parameter for exploration vs exploitation
      elapsed_time <- system.time({
        results <- BayesianOptimization(
          FUN = function(nu, gamma, kernel, number_features) {
            modelTuning(
              model = model,
              activity = activity,
              feature_data = feature_data,
              nu = nu,
              kernel = kernel,
              gamma = gamma,
              number_features = number_features,
              validation_proportion = validation_proportion,
              balance = balance
            )
          },
          bounds = bounds,
          init_points = 10,
          n_iter = 20,
          acq = "ucb",
          kappa = 2.576
        )
      })

      # Clean up and reset to sequential processing
      gc()
      plan(sequential)

      # Store results for this activity
      results_stored[[activity]] <- save_best_params(
        data_name = dataset_name,
        model_type = model,
        activity = activity,
        elapsed_time = elapsed_time,
        results = results
      )
    }

    # Save results for this model type
    save_results(
      results_stored, 
      file.path(base_path, "Output", "Tuning", 
                paste0(dataset_name, "_", model, "_hyperparmaters.csv"))
    )
  }
}



# Multiclass model tuning -------------------------------------------------
if (!file.exists(multi_hyperparam_file)) {
  print("Beginning optimization for multiclass models.")
  best_params_list <- list()

  # Define different behaviour column groupings
  behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")

  # Load feature data for multiclass optimization
  feature_data <- fread(
    file.path(base_path, "Data", "Feature_data", 
              paste0(dataset_name, "_multi_features.csv"))
  ) %>%
    as.data.table()

  # Optimize for each behaviour grouping
  for (behaviours in behaviour_columns) {
    print(paste("Tuning multiclass model for behaviour column:", behaviours))

    # Prepare data for current behavior grouping
    multiclass_data <- feature_data %>%
      select(-(setdiff(behaviour_columns, behaviours))) %>%
      rename("Activity" = !!sym(behaviours))

    if (behaviours == "GeneralisedActivity") {
      multiclass_data <- multiclass_data %>% filter(!Activity == "")
    }

    # Set up parallel processing
    plan(multisession, workers = detectCores() - 1)

    # Perform Bayesian optimization for multiclass model
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

    # Store results for this grouping
    results_stored[[activity]] <- save_best_params(
      data_name = dataset_name,
      model_type = "Multi",
      activity = behaviours,
      elapsed_time = elapsed_time,
      results = results
    )
  }

  plan(sequential)
  # Save all multiclass results
  save_results(best_params_list, multi_hyperparam_file)
} else {
  print("Multi-class parameters have already been tuned and saved")
}
