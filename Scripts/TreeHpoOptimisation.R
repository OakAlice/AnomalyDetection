#| Hyperparameter Optimisation Script for Tree types

model_type <- c("OCC", "Binary") # "Multi"

# Define bounds -----------------------------------------------------------
OCC_bounds <- list(
                  n_trees = c(50, 200),
                  max_depth = c(5, 15)
                )

Binary_bounds <- list(
                    max_depth = c(3, 20),
                    min_samples_leaf = c(5, 50),
                    min_samples_split = c(10, 100)
                  )

Multi_bounds <- list(
        n_trees = c(100, 500),          # Standard range for random forests
        mtry = c(floor(sqrt(p)), p/3),  # p is number of features
        min_samples_leaf = c(1, 20),    # Can be smaller due to ensemble
        max_depth = c(10, 30),          # Deeper trees work well in RF
        max_features = c(0.2, 0.8),     # Traditional RF feature sampling
        sample_fraction = c(0.5, 1.0)   # Bagging fraction
    )

# Load in data ------------------------------------------------------------
dichotomous_feature_data <- fread(file.path(base_path, "Data", "Feature_data", 
                              paste0(dataset_name, "_other_features.csv"))) %>%
                              select(-GeneralisedActivity, -OtherActivity) %>%
                              as.data.table()

multiclass_feature_data <- fread(file.path(base_path, "Data", "Feature_data", 
                                paste0(dataset_name, "_other_features.csv"))) %>%
                                as.data.table()


# OCC model tuning (with Isolation Forest) --------------------------------
if ("OCC" %in% model_type){
  results_stored <- list()
  for (activity in target_activities) {
    print(paste("Tuning 1-class model for activity:", activity))
    
    # Set up parallel processing
    # plan(multisession, workers = availableCores() - 1)
    
      elapsed_time <- system.time({
        results <- BayesianOptimization(
          FUN = function(n_trees, nodesize, max_depth, contamination) {
            OCCModelTuningRF(
              model = "OCC",
              activity = activity,
              feature_data = dichotomous_feature_data,
              n_trees = n_trees,
              max_depth = max_depth,
              validation_proportion = validation_proportion,
              balance = balance
            )
          },
          bounds = OCC_bounds,
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
        save_best_params_RF(
          data_name = as.character(dataset_name),
          model_type = "OCC",
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
  }
  
  save_results(
    results_stored, 
    file.path(base_path, "Output", "Tuning", ML_method,
              paste0(dataset_name, "_OCC_hyperparmaters.csv")))
}
  
# Binary model tuning (with Decision Tree) --------------------------------
if ("Binary" %in% model_type){
  results_stored <- list()
  for (activity in target_activities) {
    print(paste("Tuning Binary model for activity:", activity))
    
    # Set up parallel processing
    # plan(multisession, workers = availableCores() - 1)
    
    elapsed_time <- system.time({
      results <- BayesianOptimization(
        FUN = function(nodesize, variables) {
          BinaryModelTuningRF(
            model = "Binary",
            activity = activity,
            feature_data = dichotomous_feature_data,
            min_samples_leaf = min_samples_leaf,
            min_samples_split = min_samples_split,
            max_depth = max_depth,
            validation_proportion = validation_proportion,
            balance = balance
          )
        },
        bounds = Binary_bounds,
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
      save_best_params_RF(
        data_name = as.character(dataset_name),
        model_type = "Binary",
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
  }
  
  save_results(
    results_stored, 
    file.path(base_path, "Output", "Tuning", ML_method,
              paste0(dataset_name, "_Binary_hyperparmaters.csv")))
}

# Multiclass model tuning -------------------------------------------------
# Define different behaviour column groupings
behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")

if ("Multi" %in% model_types){
  # Optimise for each behaviour grouping at a time
  for (behaviours in behaviour_columns) {
    if(file.exists(file.path(base_path, "Output", "Tuning", 
                             paste0(dataset_name, "_", model, "_", behaviours, "_hyperparmaters.csv")))){
      print(paste0("models have been tuned for multiclass ", behaviours, " model already."))
    } else {
      print(paste("Tuning multiclass model for behaviour column:", behaviours))
      
      # Prepare data for current grouping
      multiclass_data <- multiclass_feature_data %>%
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
            bounds = Multi_bounds,
            init_points = 5,
            n_iter = 10,
            acq = "ucb",
            kappa = 2.576 
          )
        })
     
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
}
