# Remove Training Information ---------------------------------------------
all_training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_other_features.csv")))

# Copy with all behaviours ------------------------------------------------
fwrite(all_training_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_all_other_features.csv")))

# Copy with only target behaviours ----------------------------------------
target_training_data <- all_training_data %>%
  filter(Activity %in% target_activities)

fwrite(target_training_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_target_other_features.csv")))

# Copy with only some behaviours ----------------------------------------
counts <- all_training_data %>% count(Activity) %>% arrange(desc(n))
some_behaviours <- counts$Activity[1:round(length(counts$Activity)*0.5,0)]
some_behaviours <- unique(c(some_behaviours, target_activities))
  
some_training_data <- all_training_data %>%
  filter(Activity %in% some_behaviours)

fwrite(some_training_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_some_other_features.csv")))
