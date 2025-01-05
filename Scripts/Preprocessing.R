# Feature Generation  ----------------------------------------------
# does all of the data
if (file.exists(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))) {
  print("feature data already generated.")
} else {
  # extract the appropriate window_length from the dictionary above
  window_length <- window_settings[[dataset_name]][["window_length"]]
  overlap_percent <- window_settings[[dataset_name]][["overlap_percent"]]

  data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_other.csv")))

  for (id in unique(data$ID)) {
    dat <- data %>%
      filter(ID == id) %>%
      arrange(Time) %>%
      mutate(ID = as.character(ID))

    feature_data <-
      generateFeatures(
        window_length,
        sample_rate,
        overlap_percent,
        raw_data = dat,
        features = features_type
      )

    # save it
    fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_other_features.csv")))
  }

  # stitch all the id feature data back together
  files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
  pattern <- paste0(dataset_name, ".*", "other")
  matching_files <- grep(pattern, files, value = TRUE)

  feature_data_list <- lapply(matching_files, read.csv)
  feature_data <- do.call(rbind, feature_data_list)

  # save this as well
  fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))
}

# Feature Generation for Test Data -----------------------------------------
if (file.exists(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))) {
  print("test features also already generated")
} else {
  test_data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")))

  overlap_percent <- window_settings[[dataset_name]][["overlap_percent"]]

  for (id in unique(test_data$ID)) {
    dat <- test_data %>%
      filter(ID == id) %>%
      arrange(Time) %>%
      mutate(ID = as.character(ID))

    test_feature_data <-
      generateFeatures(
        window_length,
        sample_rate,
        overlap_percent,
        raw_data = dat,
        features = features_type
      )

    # save it
    fwrite(test_feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_test_", window_length, "_features.csv")))
  }

  # stitch all the id feature data back together 
  files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
  pattern <- paste0(dataset_name, ".*", "test", ".*", window_length)
  matching_files <- grep(pattern, files, value = TRUE)

  feature_data_list <- lapply(matching_files, read.csv)
  feature_data <- do.call(rbind, feature_data_list)

  # save this as well
  fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
}
