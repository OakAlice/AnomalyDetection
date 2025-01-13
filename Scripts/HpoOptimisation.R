#| Hyperparameter Optimisation Script
#|
#| This script performs Bayesian optimization to find optimal hyperparameters for:
#| One-Class Classification (OCC) models and Binary Classification models
#| as welulti-class Classification models
#| Saves optimization results to CSV files in the Output/Tuning directory.

# Define parameter bounds for Bayesian Optimization
#' nu: Controls training error/support vector ratio (0.01 to 0.5)
#' gamma: RBF kernel parameter (0.0001 to 1)
#' kernel: Kernel type (1=linear, 2=radial, 3=polynomial)
#'  number_features: Number of features to select (5 to 100)
if (ML_method == "SVM"){
   bounds <- list(
    nu = c(0.01, 0.5),
    gamma = c(0.001, 1),
    kernel = c(1, 2, 3),
    number_features = c(5, 100)
  ) 
} else if (ML_method == "Tree"){
  bounds <- list(
    n_trees = c(100, 1000),
    mtry = c(1, 200),
    nodesize = c(5, 20)
  )
}

# Dichotomous Model tuning master call --------------------------------------
# Can currently do binary and OCC, will expand to multi-class as well

model_types <- "Binary" # c("OCC", "Binary")

feature_data <- fread(
  file.path(base_path, "Data", "Feature_data", 
            paste0(dataset_name, "_other_features.csv"))
) %>%
  select(-GeneralisedActivity, -OtherActivity) %>%
  as.data.table()

results_stored <- list()

for (model in model_types){
  for (activity in target_activities) {
    print(paste("Tuning", model, "model for activity:", activity))
    
    # Set up parallel processing
    # plan(multisession, workers = availableCores() - 1)
    
    
    if (ML_method == "SVM"){
      
      # Perform Bayesian optimization
      elapsed_time <- system.time({
        results <- BayesianOptimization(
          FUN = function(nu, gamma, kernel, number_features) {
            dichotomousModelTuningSVM(
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
      
    } else if (ML_method == "RF"){
    
      elapsed_time <- system.time({
        results <- BayesianOptimization(
          FUN = function(nodesize, mtry, n_trees) {
            dichotomousModelTuningRF(
              model = model,
              activity = activity,
              feature_data = feature_data,
              nodesize = nodesize,
              # mtry = mtry,
              # n_trees = n_trees,
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
      
    }
    
    # Clean up and reset to sequential processing
    gc()
    # plan(sequential)
    
    # Store results for this activity
    result <- tryCatch(
      save_best_params(
        data_name = as.character(dataset_name),
        model_type = as.character(model),
        activity = as.character(activity),
        elapsed_time = elapsed_time,
        results = results
      ),
      error = function(e) {
        message("Error in save_best_params: ", e$message)
        return(NULL)
      }
    )
    
    # Add result to results_stored list if valid
    if (!is.null(result)) {
      results_stored[[activity]] <- result
    } else {
      message("Skipping activity ", activity, " due to error.")
    }
  }
  
  save_results(
    results_stored, 
    file.path(base_path, "Output", "Tuning", 
              paste0(dataset_name, "_", model, "_hyperparmaters.csv")))

}



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
  for (behaviours in behaviour_columns) {
    if(file.exists(file.path(base_path, "Output", "Tuning", 
                             paste0(dataset_name, "_", model, "_", behaviours, "_hyperparmaters.csv")))){
      print(paste0("models have been tuned for multiclass ", behaviours, " model already."))
    } else {
    print(paste("Tuning multiclass model for behaviour column:", behaviours))

    # Prepare data for current grouping
    multiclass_data <- feature_data %>%
      select(-(setdiff(behaviour_columns, behaviours))) %>%
      rename("Activity" = !!sym(behaviours))

    if (behaviours == "GeneralisedActivity") {
      multiclass_data <- multiclass_data %>% filter(!Activity == "")
    }

    # Set up parallel processing
    # plan(multisession, workers = availableCores() - 1)

    # Perform Bayesian optimization for multiclass model
    
    if (ML_method == "SVM"){
      
    elapsed_time <- system.time({
      results <- BayesianOptimization(
        FUN = function(nu, gamma, kernel, number_features) {
          multiclassModelTuningSVM(
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
    
    } else if (ML_method == "RF"){
      
      elapsed_time <- system.time({
        results <- BayesianOptimization(
          FUN = function(n_trees, mtry, min_node_size, sample_fraction) {
            multiclassModelTuningRF(
              model = "Multi",
              multiclass_data = multiclass_data,
              n_trees = n_trees, 
              mtry = mtry, 
              min_node_size = min_node_size, 
              sample_fraction = sample_fraction,
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
      
    }

     gc()
    # plan(sequential)
    
    # Store results for this grouping
    results_stored <- save_best_params(
      data_name = dataset_name,
      model_type = "Multi",
      activity = behaviours,
      elapsed_time = elapsed_time,
      results = results
    )
  }
    
  # Save all multiclass results
    results_df <- rbind(results_stored)
    fwrite(results_df, file.path(base_path, "Output", "Tuning", 
              paste0(dataset_name, "_Multi_", behaviours, "_hyperparmaters.csv")),  row.names = FALSE)

}

