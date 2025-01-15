#| Hyperparameter Optimisation Script for Tree types

model_type <- c("OCC", "Binary", "Multi")

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
        min_samples_split = c(10, 100),
        min_samples_leaf = c(5, 50),
        max_depth = c(10, 30) 
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
results_stored <- list()
if ("OCC" %in% model_type){
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
      features <- paste(unique(unlist(results$Pred[[which(results$History$Value == results$Best_Value)[1]]])), collapse = ", ")
      
      summarised_results <- data.frame(
        data_name = dataset_name,
        model_type = model_type,
        behaviour_or_activity = activity,
        elapsed = as.numeric(elapsed_time[3]),
        system = as.numeric(elapsed_time[2]),
        user = as.numeric(elapsed_time[1]),
        n_trees = results$Best_Par["n_trees"],
        max_depth = results$Best_Par["max_depth"],
        Best_Value = results$Best_Value,
        Selected_Features = features
      )
    
    # Add result to results_stored list if valid
    if (!is.null(summarised_results)) {
      results_stored[[activity]] <- summarised_results
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
results_stored <- list()
if ("Binary" %in% model_type){
  for (activity in target_activities) {
    print(paste("Tuning Binary model for activity:", activity))
    
    # Set up parallel processing
    # plan(multisession, workers = availableCores() - 1)
    
    elapsed_time <- system.time({
      results <- BayesianOptimization(
        FUN = function(min_samples_leaf, min_samples_split, max_depth) {
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
    features <- paste(unique(unlist(results$Pred[[which(results$History$Value == results$Best_Value)[1]]])), collapse = ", ")
    
    summarised_results <- data.frame(
      data_name = dataset_name,
      model_type = model_type,
      behaviour_or_activity = activity,
      elapsed = as.numeric(elapsed_time[3]),
      system = as.numeric(elapsed_time[2]),
      user = as.numeric(elapsed_time[1]),
      min_samples_split = results$Best_Par["min_samples_split"],
      min_samples_leaf = results$Best_Par["min_samples_leaf"],
      max_depth = results$Best_Par["max_depth"],
      Best_Value = results$Best_Value,
      Selected_Features = features
    )
    
    # Add result to results_stored list if valid
    if (!is.null(summarised_results)) {
      results_stored[[activity]] <- summarised_results
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

if ("Multi" %in% model_type){
  # Optimise for each behaviour grouping at a time
  for (behaviours in behaviour_columns) {
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
            FUN = function(min_samples_split, min_samples_leaf, max_depth) {
              multiclassModelTuningRF(
                model = "Multi",
                multiclass_data = multiclass_data,
                
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                max_depth = max_depth,
                
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
      features <- paste(unique(unlist(results$Pred[[which(results$History$Value == results$Best_Value)[1]]])), collapse = ", ")
      
      results <- data.frame(
        data_name = dataset_name,
        model_type = "Multi",
        behaviour_or_activity = behaviours,
        elapsed = as.numeric(elapsed_time[3]),
        system = as.numeric(elapsed_time[2]),
        user = as.numeric(elapsed_time[1]),
        min_samples_split = results$Best_Par["min_samples_split"],
        min_samples_leaf = results$Best_Par["min_samples_leaf"],
        max_depth = results$Best_Par["max_depth"],
        Best_Value = results$Best_Value,
        Selected_Features = features
      )
      
      # Save all multiclass results
    fwrite(results, file.path(base_path, "Output", "Tuning", ML_method,
                                 paste0(dataset_name, "_Multi_", behaviours, "_hyperparmaters.csv")),  row.names = FALSE)
    
  }
}

# Read all multi files together ----------------------------------------
tune_files <- list.files(file.path(base_path, "Output", "Tuning", ML_method), pattern = paste0(dataset_name, "_Multi_.*\\.csv$"), full.names = TRUE)
multi_tuning <- rbindlist(
  lapply(tune_files, function(file) {
    df <- fread(file)
    return(df)
  }),
  use.names = TRUE
)
fwrite(multi_tuning, file.path(base_path, "Output", "Tuning", ML_method, paste0(dataset_name, "_Multi_hyperparmaters.csv")))




