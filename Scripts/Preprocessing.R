# Preprocessing -----------------------------------------------------------
# choose behaviours, window lengths, and generate features

base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
source(file.path(base_path, "Scripts", "SetUp.R"))


# Specify Variables -------------------------------------------------------
# look at the pdf file and then specify details below
target_activities <- c("swimming", "moving", "still", "chewing")
overlap_percent <- 0

# specify the window lengths for each behaviour
behaviour_lengths <- list(Vehkaoja_Dog = list(
                                "multi" = 2.5,
                                "Walking" = 4.5,
                                "Shaking" = 1.5,
                                "Eating" = 1.5,
                                "Lying chest" = 1
                              ),
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
    new_column_data <- renameColumns(data, dataset_name)
    fwrite(new_column_data, file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
  }
}

# Feature Generation ------------------------------------------------------
for (activity in target_activities){
  
  if (file.exists( file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", activity, "_other_features.csv")))) {
    feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", activity, "_other_features.csv")))
    
  } else {
    
    # extract the appropriate window_length from the dictionary above
    window_length <- behaviour_lengths[[dataset_name]][[activity]]
    
    data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_other.csv")))
      
      for (id in unique(data$ID)) {
        dat <- data %>% filter(ID == id) %>% arrange(Time) %>% mutate(row_id = row_number())
        
        # extract the relevant rows to avoid over processing
        window_samples <- window_length*sample_rate
        subset_data <- selectRelevantData(dat, activity, window_samples)
        
        feature_data <-
          generateFeatures(
            window_length,
            sample_rate,
            overlap_percent,
            raw_data = subset_data,
            features = features_type
          )
        
        # save it 
        fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", id, "_", activity, "_features.csv")))
      }
      
      # stitch all the id feature data back together
      files <- list.files(file.path(base_path, "Data/Feature_data"), pattern = "*.csv", full.names = TRUE)
      pattern <- paste0(dataset_name, ".*", type)
      matching_files <- grep(pattern, files, value = TRUE)
      
      feature_data_list <- lapply(matching_files, read.csv)
      feature_data <- do.call(rbind, feature_data_list)
      
      # save this as well
      fwrite(feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_", type, "_features.csv")))
    }
  }