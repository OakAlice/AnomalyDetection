## Isolation Forest

# load in the data
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"

data_other <- fread(file.path(base_path, "Data", "Hold_out_test", "Vehkaoja_2018_other.csv")) %>%
  mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                           "Stationary",  Activity))

data_test <- fread(file.path(base_path, "Data", "Hold_out_test", "Vehkaoja_2018_test.csv")) %>%
  mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                           "Stationary",  Activity))

# process into features
features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "auto", "zero", "fft")
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
activity <- "Galloping"



data_other_processed <- process_data(na.omit(data_other), features_list, window_length = 1, 
                                        overlap_percent = 50, down_Hz = 100, 
                                        feature_normalisation = "Standardisation")


validation_data <- data_other_processed %>%
  filter(ID %in% unique(data_other_processed$ID)[1:5])
validation_data_features <- validation_data %>% select(-Timestamp, -ID, -Activity)

training_data <- data_other_processed %>%
  filter(ID %in% unique(data_other_processed$ID)[6:length(unique(data_other_processed$ID))]) %>%
  filter(Activity == activity)
training_data_features <- training_data %>% select(-Timestamp, -ID, -Activity)


#install.packages("isotree")
#library(isotree)


training_feature_data <- as.matrix(training_data_features)
validation_feature_data <- as.matrix(validation_data_features)

iso_forest <- isolation.forest(
  data = training_feature_data,
  ntrees = 100,      # Number of trees to build (default is 100)
  sample_size = 1000, # Subsampling size (default is 256)
  ndim = 3,          # Number of dimensions to randomly sample at each node
  prob_pick_avg_gain = TRUE # Split by average gain
  #max_depth = ceiling(log2(10))
)

# Predict anomaly scores
training_anomaly_scores <- predict(iso_forest, training_feature_data, type = "score") %>%
  data.frame() %>% 
  rename(Anomaly_Score = 1)

# Renaming the anomaly scores column for the validation data
validation_anomaly_scores <- predict(iso_forest, validation_feature_data, type = "score") %>%
  data.frame() %>% 
  rename(Anomaly_Score = 1)

# stitch them back onto the dataset
training_data_predictions <- cbind(training_data, training_anomaly_scores)
validation_data_predictions <- cbind(validation_data, validation_anomaly_scores)

# define a threshold # in this case, 90% sure
threshold_90 <- quantile(training_anomaly_scores$Anomaly_Score, 0.90)

# find all the validation data that falls within this category
validation_data_predictions <- validation_data_predictions %>%
  arrange(Anomaly_Score) %>%
  mutate(Predicted_Anomaly = ifelse(Anomaly_Score > threshold_90, TRUE, FALSE),
         Actual_Anomaly = ifelse(Activity == activity, FALSE, TRUE))

# summarise it 
summary <- validation_data_predictions %>% group_by(Predicted_Anomaly, Activity) %>% count()

# plot it
ggplot(validation_data_predictions, aes(x = Anomaly_Score, fill = Activity)) +
  geom_histogram(binwidth = 0.01) +
  labs(title = "Frequency Plot of Anomaly Scores",
       x = "Anomaly Score",
       y = "Frequency") +
  theme_minimal()

# plot again
ggplot(validation_data_predictions, aes(x = Anomaly_Score, fill = Actual_Anomaly, color = Actual_Anomaly)) +
  #geom_density(alpha = 0.4, size = 0.1) +
  geom_histogram(binwidth = 0.01, alpha = 0.4, position = "identity", color = "black", size = 0.2) +
  geom_vline(xintercept = threshold_90, linetype = "dashed", color = "black", size = 1) +
  labs(title = "Frequency Distribution of Anomaly Scores with 90% Threshold",
       x = "Anomaly Score",
       y = "Density") +
  theme_minimal()



validation_data_predictions <- validation_data_predictions %>%
  filter(Activity == activity)
