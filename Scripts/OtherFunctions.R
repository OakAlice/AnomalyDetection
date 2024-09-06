# Other functions

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
  hyperparameters_df <- data.frame(feature_set = character(0))
  
  # Iterate over each feature set in the hyperparameter list
  for (feature_set in names(feature_hyperparameters_list)) {
    feature_hyperparameters <- feature_hyperparameters_list[[feature_set]]
    
    # Create all combinations of the hyperparameters
    all_combinations <- expand.grid(feature_hyperparameters)
    
    # Add a column for the feature set
    all_combinations$feature_set <- feature_set
    
    # Bind the new combinations to the growing hyperparameters dataframe
    hyperparameters_df <- dplyr::bind_rows(hyperparameters_df, all_combinations)
  }
  
  # Merge the hyperparameters with the extended options dataframe
  extended_options_df2 <- merge(
    extended_options_df,
    hyperparameters_df[hyperparameters_df$feature_set %in% unique(extended_options_df$feature_sets), ],
    by.x = "feature_sets", by.y = "feature_set",
    all = TRUE
  )
  
  return(extended_options_df2)
}
