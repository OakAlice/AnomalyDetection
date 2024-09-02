# One vs All Binary Classification

#install.packages("caret")
library(caret)
library(randomForest)

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

data_other_processed <- process_data(na.omit(data_other), features_list, window_length = 1, 
                                     overlap_percent = 50, down_Hz = 100, 
                                     feature_normalisation = "Standardisation")

# create the specific datasets 
# choose the specific behaviour for this model
activity <- "Galloping"

validation_data <- data_other_processed %>%
  filter(ID %in% unique(data_other_processed$ID)[1:5]) %>%
  mutate(Activity = ifelse(Activity == activity, "A", "B"))
validation_data_features <- validation_data %>% select(-Timestamp, -ID, -Activity)

training_data <- data_other_processed %>%
  filter(ID %in% unique(data_other_processed$ID)[6:length(unique(data_other_processed$ID))]) %>%
  mutate(Activity = ifelse(Activity == activity, "A", "B"))
training_data_features <- training_data %>% select(-Timestamp, -ID, -Activity)

# Build the model for a binary SVM
cost_value <- 0.9
kernel_shape <- "radial"

binary_SVM <- svm(
  x = training_data_features,  # features)
  y = as.factor(training_data$Activity),  # Response variable (binary target)
  type = 'C-classification',  # Standard binary classification
  kernel = kernel_shape,  # Specify the kernel ('linear', 'polynomial', 'radial', etc.)
  cost = cost_value,  # Regularization parameter
  scale = TRUE  # Whether to scale the data
)

# now try it with a random forest
binary_RF <- randomForest::randomForest(
  x = training_data_features, 
  y = as.factor(training_data$Activity), 
  ntree = 1000, 
  importance = TRUE)


# Predict on validation data
predictions <- predict(binary_SVM, validation_data_features)
predictions <- predict(binary_RF, validation_data_features)


# Evaluate the performance using a confusion matrix
confusion_matrix <- table(Predicted = predictions, Actual = validation_data$Activity)
print(confusion_matrix)

# Optional: Calculate additional performance metrics
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy, 4)))




# Evaluate the model on validation data
predictions <- predict(model, validation_data_features)
confusionMatrix(predictions, as.factor(validation_data$Activity))

# Print model summary
print(model)