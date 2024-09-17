# ---------------------------------------------------------------------------
# Testing functions
# ---------------------------------------------------------------------------

format_final_data <- function(data,
                              number_trees = NULL, number_features = NULL,
                              specific_features = NULL){
  
    potential_features <- select_potential_features(data, threshold = 0.9)
    features_and_columsn <- c(potential_features, "Activity", "Time", "ID")
    data <- data[, ..features_and_columsn]
    data <- data[complete.cases(data), ]
    
    RF_features <- RF_feature_selection(data = data, 
                                        target_column = "Activity", 
                                        n_trees = number_trees, 
                                        number_features = number_features)
    
    top_features <- RF_features$Selected_Features$Feature[1:number_features]
    all_columns <- c(top_features, label_columns)
    selected_feature_data <- data[, ..all_columns]
    
  return(selected_feature_data)
}
