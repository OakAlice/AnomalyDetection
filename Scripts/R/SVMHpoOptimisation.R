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
bounds <- list(
  nu = c(0.01, 0.5),
  gamma = c(0.001, 1),
  kernel = c(1, 2, 3),
  number_features = c(5, 100)
) 

for (training_set in training_sets){
  tryCatch({
    # Dichotomous Model tuning master call --------------------------------------
    # Can currently do binary and OCC, will expand to multi-class as well

  model_types <- c("OCC", "Binary")
  
  feature_data <- fread(
    file.path(base_path, "Data", "Feature_data", 
              paste0(dataset_name, "_other_features.csv"))
  ) %>%
    select(-GeneralisedActivity, -OtherActivity) %>%
    as.data.table()
  
  results_stored <- list()
  
      for (model in model_types){
        for (activity in target_activities) {
          tryCatch({
            print(paste("Tuning", model, "model for activity:", activity, "from set", training_set))
            
            # Set up parallel processing
            # plan(multisession, workers = availableCores() - 1)
            
            # optimise
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
                  init_points = 5,
                  n_iter = 10,
                  acq = "ucb",
                  kappa = 2.576
                )
              })
            
            # Clean up and reset to sequential processing
            gc()
            # plan(sequential)
            
            # Store results for this activity
            # rewrite this code to be better - i.e., handle ML method inside of function
            result <- tryCatch({
                save_best_params(
                  data_name = as.character(dataset_name),
                  model_type = as.character(model),
                  activity = as.character(activity),
                  elapsed_time = elapsed_time,
                  results = results
                )
            }, error = function(e) {
              message("Error in save_best_params: ", e$message)
              return(NULL)
            })
            
            # Add result to results_stored list if valid
            if (!is.null(result)) {
              results_stored[[activity]] <- result
            } else {
              message("Skipping activity ", activity, " due to error.")
            }
            
          }, error = function(e) {
            message(paste("Error processing activity:", activity, "for model:", model, "\nError message:", e$message))
            # Continue to next activity
          })
        }
        
        save_results(
          results_stored, 
          file.path(base_path, "Output", "Tuning", ML_method,
                    paste0(dataset_name, "_", training_set, "_", model, "_hyperparmaters.csv")))
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
  results_stored <- list()  # Initialize results list
  
  for (behaviours in behaviour_columns) {
    tryCatch({
      print(paste("Tuning multiclass model for behaviour column:", behaviours, "from dataset", training_set))

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
      
      gc()
      # plan(sequential)
      
      # Store results for this grouping
      results_stored[[behaviours]] <- save_best_params(
        data_name = dataset_name,
        model_type = "Multi",
        activity = behaviours,
        elapsed_time = elapsed_time,
        results = results
      )
      
    }, error = function(e) {
      message(paste("Error processing behaviour:", behaviours, "\nError message:", e$message))
      message(print(results))
      # Continue to next behaviour
    })
  }
    
    # Save all multiclass results if we have any
    if (length(results_stored) > 0) {
      results_df <- rbind(results_stored)
      fwrite(results_df, file.path(base_path, "Output", "Tuning", 
                paste0(dataset_name, "_", training_set, "_Multi_hyperparmaters.csv")),  row.names = FALSE)
    }

  }, error = function(e) {
    message(paste("Error processing training set:", training_set, "\nError message:", e$message))
    # Continue to next training set
  })
}
