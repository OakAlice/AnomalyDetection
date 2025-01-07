# Assorted functions

# ensure a directory exists
ensure.dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

# Function to apply column selection changes to both training and testing data
update_feature_data <- function(data, multi) {
  cols_to_remove <- c("Activity", "GeneralisedActivity", "OtherActivity")
  # classes to remove logic
  if (multi == "OtherActivity") {
    col_to_rename <- "OtherActivity"
  } else if (multi == "GeneralisedActivity") {
    col_to_rename <- "GeneralisedActivity"
  } else if (multi == "Activity") {
    col_to_rename <- "Activity"
  }

  data <- data %>%
    select(-(setdiff(cols_to_remove, col_to_rename))) %>%
    rename(Activity = col_to_rename)

  return(data)
}

#' Adjust activity labels based on model type
#' @param data Data.table containing feature data
#' @param model Model type ("OCC" or "Binary")
#' @param activity Target activity to classify
#' @return Data.table with adjusted activity labels
adjust_activity <- function(data, model, activity) {
  # Ensure the input is a data.table
  data <- data.table::as.data.table(data)
  
  # Adjust the Activity column based on the model type
  data[, Activity := ifelse(Activity == activity, activity, "Other")]
  
  data <- data[Activity %in% c(activity, "Other")]
  
  return(data)
}


#' Ensure target activity is represented in validation data
#' @param validation_data Data.table containing validation set
#' @param model Model type
#' @param retries Number of attempts to create valid split
#' @return Data.table with validated split
ensure_activity_representation <- function(validation_data, model, retries = 10) {
  retry_count <- 0
  while (sum(validation_data$Activity == activity) == 0 && retry_count < retries) {
    retry_count <- retry_count + 1
    message(activity, " not represented in validation fold. Retrying... (Attempt ", retry_count, ")")
    test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
    validation_data <- feature_data[ID %in% test_ids]
    validation_data <- adjust_activity(validation_data, model, activity)
  }
  if (retry_count == retries) stop("Unable to find a valid validation split after ", retries, " attempts.")
  return(validation_data)
}

#' Balance dataset by undersampling majority classes
#' @param data Data.table to balance
#' @return Balanced data.table
balance_data <- function(data, activity) {
  activity_count <- data[data$Activity == activity, .N] / length(unique(data$Activity))
  data[, .SD[sample(.N, min(.N, activity_count))], by = Activity]
}

#' Split data into training and validation sets and format approporiately
#' @param model Model type ("OCC", "Binary", or "Multi")
#' @param activity Target activity
#' @param balance Balancing strategy
#' @param feature_data Input feature data
#' @param validation_proportion Proportion for validation set
#' @return List containing training and validation datasets
split_data <- function(model, activity, balance, feature_data, validation_proportion) {
  # Ensure feature_data is a data.table
  setDT(feature_data)
  
  unique_ids <- unique(feature_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * validation_proportion))
  
  training_data <- feature_data[!ID %in% test_ids]
  validation_data <- feature_data[ID %in% test_ids]
  
  # Balance validation and training data
  if (model == "OCC") {
    training_data <- training_data[training_data$Activity == activity, ]
    # Apply balancing only to validation data if needed
    if (balance == "stratified_balance") {
      validation_data <- balance_data(validation_data, activity)
    }
  } else if (model == "Binary") {
    if (balance == "stratified_balance") {
      validation_data <- balance_data(validation_data, activity)
      training_data <- balance_data(training_data, activity)
    }
  } else if (model == "Multi"){
    # not sure whether I should add balancing logic here...
    validation_data <- validation_data
    training_data <- training_data
  }
  
  if (!model == "Multi"){
    # Adjust training and validation data labels (e.g., grouping non-target to "Other")
    training_data <- adjust_activity(training_data, model, activity)
    validation_data <- adjust_activity(validation_data, model, activity)
    
    # Retry logic if the target activity is not represented
    validation_data <- ensure_activity_representation(validation_data, model)
  } 
  
  if (model == "OCC") {
    training_data <- training_data[training_data$Activity == activity, ]
  }
  
  return(list(training_data = training_data, validation_data = validation_data))
}


save_best_params <- function(data_name, model_type, activity, elapsed_time, results) {
  
  features <- paste(unique(unlist(results$Pred[[which(results$History$Value == results$Best_Value)[1]]])), collapse = ", ")
  
  results <- data.frame(
    data_name = data_name,
    model_type = model_type,
    behaviour_or_activity = activity,
    elapsed = as.numeric(elapsed_time[3]),
    system = as.numeric(elapsed_time[2]),
    user = as.numeric(elapsed_time[1]),
    nu = results$Best_Par["nu"],
    gamma = results$Best_Par["gamma"],
    kernel = results$Best_Par["kernel"],
    number_trees = ifelse(!is.na(results$Best_Par["number_trees"]), results$Best_Par["number_trees"], NA),
    number_features = ifelse(!is.na(results$Best_Par["number_features"]), results$Best_Par["number_features"], NA),
    Best_Value = results$Best_Value,
    Selected_Features = features
  )
  return(results) 
}

save_results <- function(results_list, file_path) {
  results_df <- rbindlist(results_list, use.names = TRUE, fill = TRUE)
  fwrite(results_df, file_path, row.names = FALSE)
}
