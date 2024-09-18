# ---------------------------------------------------------------------------
# Assorted functions
# ---------------------------------------------------------------------------

source(file.path("C:/Users/oaw001/Documents/AnomalyDetection", "Scripts", "UserInput.R"))

# ensure a directory exists
ensure.dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

# increase hyperparameter options in options_df ####
# bind this with the normal parameter options
create_extended_options <- function(model_hyperparameters_list, options_df) {
  # Expand hyperparameters for each model
  all_hyperparameters_df <- expand_hyperparameters(model_hyperparameters_list)
  
  # Merge only the model_architectures that appear in options_df
  extended_options_df <- merge(
    options_df,
    all_hyperparameters_df[all_hyperparameters_df$kernel %in% unique(options_df$kernel), ],
    by = "kernel",
    all = TRUE
  )
  
  return(extended_options_df)
}

# also important 
expand_hyperparameters <- function(model_hyperparameters_list) {
  hyperparameters_df <- data.frame(kernel = character(0))
  
  # Iterate over each model hyperparameters
  for (kernel_name in names(model_hyperparameters_list)) {
    model_hyperparameters <- model_hyperparameters_list[[kernel_name]]
    param_names <- names(model_hyperparameters)
    all_combinations <- expand.grid(model_hyperparameters)
    
    all_combinations$kernel <- kernel_name
    
    # Merge with existing hyperparameters dataframe
    hyperparameters_df <- dplyr::bind_rows(hyperparameters_df, all_combinations)
  }
  
  return(hyperparameters_df)
}

create_extended_options2 <- function(feature_hyperparameters_list, extended_options_df) {
  
  # Initialize an empty dataframe for storing expanded hyperparameters
  hyperparameters_df <- data.frame(feature_selection_method = character(0))
  
  # Iterate over each feature set in the hyperparameter list
  for (feature_selection_method in names(feature_hyperparameters_list)) {
    feature_hyperparameters <- feature_hyperparameters_list[[feature_selection_method]]
    
    # Create all combinations of the hyperparameters
    all_combinations <- expand.grid(feature_hyperparameters)
    
    # Add a column for the feature set
    all_combinations$feature_selection_method <- feature_selection_method
    
    # Bind the new combinations to the growing hyperparameters dataframe
    hyperparameters_df <- dplyr::bind_rows(hyperparameters_df, all_combinations)
  }
  
  # Merge the hyperparameters with the extended options dataframe
  extended_options_df2 <- merge(
    extended_options_df,
    hyperparameters_df[hyperparameters_df$feature_selection_method %in% unique(extended_options_df$feature_selection_method), ],
    by.x = "feature_selection_method", by.y = "feature_selection_method",
    all = TRUE
  )
  
  return(extended_options_df2)
}




# ---------------------------------------------------------------------------
# Generate dataframe with all combinations of options
# ---------------------------------------------------------------------------

expand_all_options <- function(model_hyperparameters_list, feature_hyperparameters_list,
                               targetActivity_options, model_options, 
                               feature_selection_method, feature_normalisation_options, 
                               nu_options, kernel_options, degree_options) {
  
  # Create initial combinations of general options
  options_df <- expand.grid(targetActivity = targetActivity_options, 
                            model = model_options,
                            feature_selection_method = feature_selection_method, 
                            feature_normalisation = feature_normalisation_options, 
                            nu = nu_options, 
                            kernel = kernel_options,
                            degree = degree_options)
  
  # Extend the options with model hyperparameters
  extended_options_df <- create_extended_options(model_hyperparameters_list, options_df)
  
  # Further extend with feature hyperparameters and convert to data.table
  extended_options_df2 <- create_extended_options2(feature_hyperparameters_list, extended_options_df) %>% setDT()
  
  return(extended_options_df2)
}



model_hyperparameters_list <- list(
  radial = list(gamma = gamma_options),
  polynomial = list(gamma = gamma_options, degree = degree_options),
  sigmoid = list(gamma = gamma_options)
)

feature_hyperparameters_list <- list(
  UMAP = list(min_dist = minimum_distance_options,
              n_neighbours = num_neighbours_options,
              metric = shape_metric_options),
  RF = list(number_trees = number_trees_options,
            number_features = number_features_options)
)



