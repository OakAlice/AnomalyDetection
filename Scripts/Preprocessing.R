# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data 2                ####
# ---------------------------------------------------------------------------
# choose behaviours, window lengths, and generate features

base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
source(file.path(base_path, "Scripts", "SetUp.R"))

# Specify Variables -------------------------------------------------------
# look at the pdf file and then specify details below
#target_activities <- c("swimming", "moving", "still", "chewing")
target_activities <- c("Walking", "Shaking", "Eating", "Lying chest")
overlap_percent <- 0

# specify the window lengths for each behaviour
window_settings <- list(Vehkaoja_Dog = list(window_length = 1, overlap_percent = 50),
                          Ladds_Seal = list(
                                "multi" = 0.5,
                                "swimming" = 1,
                                "scratch" = 1,
                                "still" = 0.5,
                                "chewing" = 0.5
                              )
)

# Adding activity category columns ----------------------------------------
# specifying categories for the behaviours so we can test different combinations
if (renamed == FALSE){
  for (condition in c("other", "test")){
    data <- fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
    new_column_data <- renameColumns(data, dataset_name, target_activities)
    fwrite(new_column_data, file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
  }
}

# Feature Generation for a specific behaviour ------------------------------
# only this key behaviour with other non-target behaviours to make mixed windows
# for (activity in target_activities){
#   
#   print(activity)
#   
#   if (file.exists( file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", activity, "_other_features.csv")))) {
#     feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", activity, "_other_features.csv")))
#     
#   } else {
#     
#     # extract the appropriate window_length from the dictionary above
#     window_length <- behaviour_lengths[[dataset_name]][[activity]]
#     
#     data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_other.csv")))
#       
#       for (id in unique(data$ID)[2:length(unique(data$ID))]) {
#         dat <- data %>% filter(ID == id) %>% arrange(Time) %>% mutate(row_id = row_number())
#         dat$ID <- as.character(dat$ID)
#         
#         # extract the relevant rows to avoid over processing
#         window_samples <- window_length*sample_rate
#         if (activity %in% unique(dat$Activity)){
#           subset_data <- selectRelevantData(dat, activity, window_samples)
#           
#           feature_data <-
#             generateFeatures(
#               window_length,
#               sample_rate,
#               overlap_percent,
#               raw_data = subset_data,
#               features = features_type
#             )
#           
#           # save it 
#           fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_", activity, "_features.csv")))
#         } else {
#           print("No target activity in this individual")
#         }
#       }
#       
#       # stitch all the id feature data back together
#       files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
#       pattern <- paste0(dataset_name, ".*", activity)
#       matching_files <- grep(pattern, files, value = TRUE)
#       
#       feature_data_list <- lapply(matching_files, read.csv)
#       feature_data <- do.call(rbind, feature_data_list)
#       
#       # save this as well
#       fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", activity, "_features.csv")))
#     }
# }

# Feature Generation  ----------------------------------------------
# does all of the data
if (file.exists( file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_other_features.csv")))) {
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_other_features.csv")))
    
  } else {
    
    # extract the appropriate window_length from the dictionary above
    window_length <- window_settings[[dataset_name]][["window_length"]]
    overlap_percent <- window_settings[[dataset_name]][["overlap_percent"]]
    
    data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_other.csv")))
    
    for (id in unique(data$ID)) {
      dat <- data %>% filter(ID == id) %>% arrange(Time) %>% mutate(ID = as.character(ID))
      
      feature_data <-
        generateFeatures(
          window_length,
          sample_rate,
          overlap_percent,
          raw_data = dat,
          features = features_type
        )
      
      # save it 
      fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_multi_features.csv")))
    }
    
    # stitch all the id feature data back together
    files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
    pattern <- paste0(dataset_name, ".*", "multi")
    matching_files <- grep(pattern, files, value = TRUE)
    
    feature_data_list <- lapply(matching_files, read.csv)
    feature_data <- do.call(rbind, feature_data_list)
    
    # save this as well
    fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv")))
  }

# Feature Generation for Test Data -----------------------------------------
for (window_length in unique(window_settings[[dataset_name]]$window_length)){
  if (file.exists( file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_", window_length, "_features.csv")))) {
    test_feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_", window_length, "_features.csv")))
    
  } else {
    
    test_data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")))
    
    overlap_percent <- window_settings[[dataset_name]][["overlap_percent"]]
    
    for (id in unique(test_data$ID)) {
      dat <- test_data %>% filter(ID == id) %>% arrange(Time) %>% 
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
    fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_", window_length, "_features.csv")))
  }
}




# extract data from existing processed data --------------------------------
# data <- fread(file.path(base_path, "Data", "Feature_data", paste0("Ladds_Seal_other_features.csv")))
# swimming_rows <- data %>% filter(Activity == "scratch") %>% count()
# swimming_data <- data %>% filter(Activity == "scratch")
# swimming_rows <- nrow(swimming_data)  # Define the number of swimming rows
# non_swimming_data <- data %>% filter(Activity != "scratch") %>% sample_n(swimming_rows)
# swimming_set <- rbind(swimming_data, non_swimming_data)
# fwrite(swimming_set, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_scratch_features.csv")))
