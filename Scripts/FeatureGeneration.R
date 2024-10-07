# ---------------------------------------------------------------------------
# Generating statistical and time series features
# ---------------------------------------------------------------------------

# Function to process data for each ID
process_id_data <- function(id_data, features_type, window_length, sample_frequency, overlap_percent) {
  
  # calculate window length and overlap
  samples_per_window <- window_length * sample_frequency
  overlap_samples <- if (overlap_percent > 0) ((overlap_percent / 100) * samples_per_window) else 0
  num_windows <- ceiling((nrow(id_data) - overlap_samples) / (samples_per_window - overlap_samples))
  
  # Function to process each window for this specific ID
  process_window <- function(i) {
    start_index <- max(1, round((i - 1) * (samples_per_window - overlap_samples) + 1))
    end_index <- min(start_index + samples_per_window - 1, nrow(id_data))
    window_chunk <- id_data[start_index:end_index, ]
    
    # extract statistical features
    if ("statistical" %in% features_type){
    statistical_features <- generate_statistical_features(window_chunk = window_chunk, down_Hz = movement_data$Frequency)
    } 
    
    # extract timeseries features and flatten
    if ("timeseries" %in% features_type){
      time_series_features <- generate_ts_features(data = window_chunk)
      single_row_features <- time_series_features %>%
        mutate(axis = c("X", "Y", "Z")) %>% 
        pivot_longer(cols = -axis, names_to = "feature", values_to = "value") %>%  # Reshape the data from wide to long format
        unite("feature_name", axis, feature, sep = "_") %>%  # Combine axis and feature name
        pivot_wider(names_from = feature_name, values_from = value)  # Reshape back to wide format with prefixed feature names
    }
    
    # Extract window identifying info
    window_info <- window_chunk %>% 
      summarise(
        Time = Time[1],
        ID = ID[1],
        Activity = as.character(
          names(sort(table(Activity), decreasing = TRUE))[1]
        )
      ) %>% ungroup()
    
    # Combine the window info, time series, and statistical features
    if (exists("window_info")) window_info <- window_info else window_info <- NULL
    if (exists("single_row_features")) single_row_features <- single_row_features else single_row_features <- NULL
    if (exists("statistical_features")) statistical_features <- statistical_features else statistical_features <- NULL
    
    window_features <- do.call(cbind, Filter(Negate(is.null), list(window_info, single_row_features, statistical_features)))
    
    return(window_features)
  }
  
  # Use lapply to process each window for the current ID
  window_features_list <- lapply(1:num_windows, process_window)
  
  # Combine all the windows for this ID into a single data frame
  features <- do.call(rbind, window_features_list)
  return(features)
}


generateFeatures <- function(window_length, sample_frequency, overlap_percent, data, normalise, features_type) {
  
  # multiprocessing   
  plan(multisession, workers = availableCores())  # Use parallel processing 
  
  # Split data by 'ID'
  data_by_id <- split(data, by = "ID")
  
  # Process each ID's data
  features_by_id <- list()
  for (id in names(data_by_id)) {
    features_by_id[[id]] <- process_id_data(id_data = data_by_id[[as.character(id)]], features_type, window_length, sample_frequency, overlap_percent)
  }
  all_features <- do.call(rbind, features_by_id)
  
  plan(sequential)  # Return to sequential execution
  
  all_features <- rbindlist(features_by_id)
  
  # Optionally normalise the feature
  #if (normalise == "z_scale") {
  #  normal_ts <- as.data.frame(scale(do.call(rbind, time_series_list)))
  #  normal_stat <- as.data.frame(scale(do.call(rbind, statistical_list)))
  #  all_features <- cbind(do.call(rbind, window_info_list), normal_ts, normal_stat)
  #}
  
  return(all_features)
}


# generate time series tsfeatures ####
generate_ts_features <- function(data){
  # Convert each column (e.g., X, Y, Z) into a list of time series
  ts_list <- list(
    X = data[["Accelerometer.X"]],
    Y = data[["Accelerometer.Y"]],
    Z = data[["Accelerometer.Z"]]
  )
  
  # Generate ts features for the window
  tryCatch({
    time_series_features <- tsfeatures(tslist = ts_list,
               features = c(
                 "acf_features", "arch_stat", "autocorr_features",
                 "crossing_points", "dist_features", "entropy",
                 "firstzero_ac", "flat_spots", "heterogeneity",
                 "hw_parameters", "hurst", "lumpiness", "stability",
                 "max_level_shift", "max_var_shift", "max_kl_shift", "nonlinearity", "pacf_features",
                 "pred_features", "scal_features", "station_features", "stl_features",
                 "unitroot_kpss", "zero_proportion"
               ),
               scale = FALSE,
               multiprocess = TRUE)
  }, error = function(e) {
    message("Error in tsfeatures: ", e$message)
    return(NA)  # Return NA in case of error
  })
  
  return(time_series_features)
}

# generate statistical features ####
# Fast Fourier Transformation based features
extract_FFT_features <- function(window_data, down_Hz) {
  n <- length(window_data)
  
  # Compute FFT
  fft_result <- fft(window_data)
  
  # Compute frequencies
  freq <- (0:(n/2 - 1)) * (down_Hz / n)
  
  # Compute magnitude
  magnitude <- abs(fft_result[1:(n/2)])
  
  # Calculate features
  mean_magnitude <- mean(magnitude)
  max_magnitude <- max(magnitude)
  total_power <- sum(magnitude^2)
  peak_frequency <- freq[which.max(magnitude)]
  
  # Return features
  return(list(Mean_Magnitude = mean_magnitude,
              Max_Magnitude = max_magnitude,
              Total_Power = total_power,
              Peak_Frequency = peak_frequency))
}


# making this faster using := which modifies in place rather than copying and modifying
generate_statistical_features <- function(window_chunk, down_Hz) {
  
  # Determine the available axes from the dataset
  available_axes <- intersect(colnames(window_chunk), all_axes) # the ones we actually have
  
  result <- data.table()
  
  window_chunk <- setDT(window_chunk)

  for (axis in available_axes) {
    axis_data <- window_chunk[[axis]]  # Extract the data for the window
    
    # Compute stats
    stats <- lapply(list(mean = mean, max = max, min = min, sd = sd), 
                    function(f) f(axis_data, na.rm = TRUE))
    
    # Assign stats to result
    result[, paste0(c("mean_", "max_", "min_", "sd_"), axis) := stats]
    
    # Calculate skewness
    result[, paste0("sk_", axis) := e1071::skewness(axis_data, na.rm = TRUE)]
    
    # Extract FFT features
    fft_features <- extract_FFT_features(axis_data, down_Hz)
    
    # Add FFT features to result as well
    result[, paste0(c("mean_mag_", "max_mag_", "total_power_", "peak_freq_"), axis) := 
             list(fft_features$Mean_Magnitude, fft_features$Max_Magnitude, 
                  fft_features$Total_Power, fft_features$Peak_Frequency)]
  }
  
  # calculate SMA, ODBA, and VDBA
  result[, SMA := sum(rowSums(abs(window_chunk[, ..available_axes]))) / nrow(window_chunk)]
  ODBA <- rowSums(abs(window_chunk[, ..available_axes]))
  result[, `:=`(
    minODBA = min(ODBA, na.rm = TRUE),
    maxODBA = max(ODBA, na.rm = TRUE)
  )]
  VDBA <- sqrt(rowSums(window_chunk[, ..available_axes]^2))
  result[, `:=`(
    minVDBA = min(VDBA, na.rm = TRUE),
    maxVDBA = max(VDBA, na.rm = TRUE)
  )]
  
  # Create all unique axis pairs
  axis_pairs <- CJ(axis1 = available_axes, axis2 = available_axes)[axis1 < axis2]
  
  # Calculate correlations for each
  axis_correlations <- axis_pairs[, {
    vec1 <- window_chunk[[axis1]]
    vec2 <- window_chunk[[axis2]]
    
    # Check for non-NA and non-zero variance
    var_vec1 <- var(vec1, na.rm = TRUE)
    var_vec2 <- var(vec2, na.rm = TRUE)
    
    if (!is.na(var_vec1) && var_vec1 != 0 && !is.na(var_vec2) && var_vec2 != 0) {
      complete_cases <- complete.cases(vec1, vec2)
      if (any(complete_cases)) {
        cor_value <- cor(vec1[complete_cases], vec2[complete_cases])
      } else {
        cor_value <- NA  # No complete pairs
      }
    } else {
      cor_value <- NA  # No variability or NA in variance
    }
    
    list(cor_value)
  }, by = .(axis1, axis2)]
  
  # Add correlations to result data.table
  for (i in seq_len(nrow(axis_correlations))) {
    result[, paste0("cor_", axis_correlations$axis1[i], "_", axis_correlations$axis2[i]) := axis_correlations$cor_value[i]]
  }

  return(result)
}

# balance the data ####
balance_data <- function(dat, threshold) {
  #dat <- processed_data
  
  # Determine counts of each 'Activity' and identify over-represented behaviors
  dat_counts <- dat %>%
    count(Activity)
  
  # For over-represented behaviors, sample the desired threshold number of rows or all if less
  dat_selected <- dat %>%
    group_by(Activity, ID) %>%
    mutate(row_number = row_number()) %>%
    ungroup() %>%
    inner_join(dat_counts, by = "Activity") %>%
    mutate(max_rows = if_else(n > threshold, threshold, n)) %>%
    filter(row_number <= max_rows) %>%
    select(-row_number, -n, -max_rows)
  
  # Combine and return
  balance_data <- dat_selected
  return(balance_data)
}



