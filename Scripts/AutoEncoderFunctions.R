# Functions for anomaly detection with autoencoder


create_windows <- function(data, window_size) {
  n <- nrow(data)
  windows <- rollapply(data, 
                       width = window_size, 
                       by = 1, 
                       align = 'left', 
                       FUN = function(x) as.data.frame(t(x)), 
                       by.column = FALSE,
                       fill = NA)
  windows <- na.omit(as.data.frame(windows))
  return(windows)
}

# create the target training data
create_format_data <- function(training_data, targetActivity, window_size){
  
  target_training_data <- training_data %>% 
    filter(Activity == targetActivity) %>%
    select(-Activity, -time, -ID) %>%
    scale()
  
  anom_training_data <- training_data %>% 
    filter(Activity != targetActivity) %>%
    select(-Activity, -time, -ID) %>% 
    scale() %>%
    as.data.frame() %>%               # Convert back to data frame
    slice(1:nrow(target_training_data))
  
  print("created the datasets")
  
  # turn them into windows
  target_windows <- create_windows(target_training_data, window_size)
  anom_windows <- create_windows(anom_training_data, window_size)
  
  
  print("converted them to windows")
  
  # convert to h2o objects
  target_windows = as.h2o(target_windows)
  anom_windows = as.h2o(anom_windows)
  
  print("converted to h2o objects")
  
  return(list(target_windows = target_windows,
              anom_windows = anom_windows))
}
