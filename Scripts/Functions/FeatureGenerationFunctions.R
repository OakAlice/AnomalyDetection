# ---------------------------------------------------------------------------
# Generating statistical and time series features
# ---------------------------------------------------------------------------

# Function to process data for each ID
processDataPerID <- function(id_raw_data, features_type, window_length, sample_rate, overlap_percent) {
  
  # Calculate window length and overlap
  samples_per_window <- window_length * sample_rate
  overlap_samples <- if (overlap_percent > 0) ((overlap_percent / 100) * samples_per_window) else 0
  num_windows <- ceiling((nrow(id_raw_data) - overlap_samples) / (samples_per_window - overlap_samples))
  
  # Function to process each window for this specific ID
  process_window <- function(i) {
    print(i)
    start_index <- max(1, round((i - 1) * (samples_per_window - overlap_samples) + 1))
    end_index <- min(start_index + samples_per_window - 1, nrow(id_raw_data))
    window_chunk <- id_raw_data[start_index:end_index, ]
    
    # Initialize output features
    window_info <- tibble(Time = NA, ID = NA, Activity = NA, GeneralisedActivity = NA, OtherActivity = NA)
    statistical_features <- tibble() 
    single_row_features <- tibble()  
    
    # Extract statistical features
    if ("statistical" %in% features_type) {
      statistical_features <- generateStatisticalFeatures(window_chunk = window_chunk, down_Hz = sample_rate)
    }
    
    # Extract timeseries features and flatten
    if ("timeseries" %in% features_type) {
      time_series_features <- tryCatch({
        generateTsFeatures(data = window_chunk)
      }, error = function(e) {
        message("Error in tsfeatures: ", e$message)
      })
      
      if (nrow(time_series_features) > 0) {
        single_row_features <- time_series_features %>%
          mutate(axis = rep(c("X", "Y", "Z"), length.out = n())) %>%
          pivot_longer(cols = -axis, names_to = "feature", values_to = "value") %>%
          unite("feature_name", axis, feature, sep = "_") %>%
          pivot_wider(names_from = feature_name, values_from = value)
      } else {
        message("No rows in time_series_features. Returning empty tibble.")
        single_row_features <- tibble(matrix(NA, nrow = 1, ncol = ncol(time_series_features))) # Fill with NAs
        colnames(single_row_features) <- colnames(time_series_features)  # Match the column names
      }
    }
    
    # Extract window identifying info
    if (nrow(window_chunk) > 0) {
      window_info <- window_chunk %>% 
        summarise(
          Time = Time[1],
          ID = ID[1],
          Activity = as.character(
            names(sort(table(Activity), decreasing = TRUE))[1]),
          GeneralisedActivity = as.character(
              names(sort(table(Activity), decreasing = TRUE))[1]), 
          OtherActivity = as.character(
                names(sort(table(Activity), decreasing = TRUE))[1])
        ) %>% ungroup()
    }
    
    # Combine the window info, time series, and statistical features
    combined_features <- bind_cols(window_info, single_row_features, statistical_features) %>% 
      mutate(across(everything(), ~replace_na(., NA)))  # Ensure all columns are present
    
    return(combined_features)
  }
  
  # Use lapply to process each window for the current ID
  window_features_list <- lapply(1:num_windows, process_window)
  
  # Combine all the windows for this ID into a single data frame
  features <- bind_rows(window_features_list)
  return(features)
}


generateFeatures <- function(window_length, sample_rate, overlap_percent, raw_data, features_type) {
  
  # multiprocessing   
  #plan(multisession, workers = availableCores())  # Use parallel processing 
  
  # Split raw_data by 'ID' # was by = "ID" before
  raw_data_by_id <- split(raw_data, raw_data$ID)
  
  # Process each ID's raw_data
  features_by_id <- list()
  for (id in unique(raw_data$ID)) {
    print(id)
    # I changed the way this was subsetted. Was previously raw_data_by_id[[as.character(id)]]
    features_by_id[[id]] <- processDataPerID(
      id_raw_data = raw_data_by_id[[id]],
      features_type,
      window_length,
      sample_rate,
      overlap_percent
    )
  }
  all_features <- do.call(rbind, features_by_id)
  
  #plan(sequential)  # Return to sequential execution
  
  all_features <- rbindlist(features_by_id)
  
  return(all_features)
}


# generate time series tsfeatures ####
generateTsFeatures <- function(data){
  # Convert each column (e.g., X, Y, Z) into a list of time series
  ts_list <- list(
    X = data[["Accelerometer.X"]],
    Y = data[["Accelerometer.Y"]],
    Z = data[["Accelerometer.Z"]]
  )
  
  # Generate ts features for the window
  # had a lot of errors and failure so wrapped in try catch
  time_series_features <- tryCatch({
    tsfeatures(
      tslist = ts_list,
      features = c(
        "acf_features", "arch_stat", "autocorr_features", "crossing_points", "dist_features",
        "entropy", "firstzero_ac", "flat_spots", "heterogeneity", "hw_parameters", "hurst",
        "lumpiness", "stability", "max_level_shift", "max_var_shift", "max_kl_shift", 
        "nonlinearity", "pacf_features", "pred_features", "scal_features", "station_features", 
        "stl_features", "unitroot_kpss", "zero_proportion"
      ),
      scale = FALSE,
      multiprocess = TRUE
    )
  }, error = function(e) {
    message("Error in tsfeatures: ", e$message)
    message("Data causing the error: ", head(data))
    return(tibble())
  })
  
  # error statement for debugging purposes
  if (nrow(time_series_features) == 0) {
    message("No features were generated. Returning empty tibble.")
  }
  
  return(time_series_features)
}

# generate statistical features ####
# Fast Fourier Transformation based features
extractFftFeatures <- function(window_data, down_Hz) {
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
generateStatisticalFeatures <- function(window_chunk, down_Hz) {
  
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
    fft_features <- extractFftFeatures(axis_data, down_Hz)
    
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
  #axis_pairs <- CJ(axis1 = available_axes, axis2 = available_axes)[axis1 < axis2]
  
  # Calculate correlations for each
  #axis_correlations <- axis_pairs[, {
  #  vec1 <- window_chunk[[axis1]]
  #  vec2 <- window_chunk[[axis2]]
    
    # Check for non-NA and non-zero variance
  #  var_vec1 <- var(vec1, na.rm = TRUE)
  #  var_vec2 <- var(vec2, na.rm = TRUE)
    
  #  if (!is.na(var_vec1) && var_vec1 != 0 && !is.na(var_vec2) && var_vec2 != 0) {
  #    complete_cases <- complete.cases(vec1, vec2)
  #    if (any(complete_cases)) {
  #      cor_value <- cor(vec1[complete_cases], vec2[complete_cases])
  #    } else {
  #      cor_value <- NA  # No complete pairs
  #    }
  #  } else {
  #    cor_value <- NA  # No variability or NA in variance
  #  }
    
  #  list(cor_value)
  #}, by = .(axis1, axis2)]
  
  # Add correlations to result data.table
  #for (i in seq_len(nrow(axis_correlations))) {
  #  result[, paste0("cor_", axis_correlations$axis1[i], "_", axis_correlations$axis2[i]) := axis_correlations$cor_value[i]]
  #}
  
  return(result)
}



# select a subset of data so I dont have to process too much
selectRelevantData <- function(dat, target_activity, window_samples) {
  
  if (target_activity %in% unique(dat$Activity)) {
    
    # Select the rows of target data (with additional windows)
    selected_rows <- dat %>%
      arrange(row_id) %>%
      mutate(
        Activity_Group = cumsum(c(TRUE, diff(Activity == target_activity) != 0)) * (Activity == target_activity),
        row_number = ifelse(Activity_Group > 0, row_number(), NA)
      ) %>%
      group_by(Activity_Group) %>%
      summarise(
        First = ceiling(first(na.omit(row_number)) - window_samples),
        Last = floor(last(na.omit(row_number)) + window_samples)
      ) %>%
      ungroup() %>%
      na.omit()
    
    new_dataframe <- selected_rows %>%
      rowwise() %>%
      mutate(Sequence = list(seq(First, Last))) %>%
      unnest(cols = c(Sequence)) %>%
      select(Activity_Group, Sequence)
    
    filtered_dat <- dat %>% 
      arrange(row_id) %>%
      filter(row_number() %in% new_dataframe$Sequence)
    
    # Grab non-target data to balance it as well
    samples <- nrow(filtered_dat)
    
    non_target_dat <- dat %>% 
      arrange(row_id) %>% 
      slice(1:samples)
    
    subset_data <- rbind(filtered_dat, non_target_dat) %>%
      arrange(row_id)
    
    return(subset_data)
  } else {
    stop("The target activity is not present in the dataset.")
  }
}

  