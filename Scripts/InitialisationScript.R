# Initialisation Script

# Set up ####
# load packages
library(data.table)
library(tidyverse)
library(future)
library(future.apply)
library(e1071)
library(h2o)
library(zoo)


# set base path
base_path <- setwd("C:/Users/oaw001/Documents/AnomalyDetection")

# load in the files
source(file.path(base_path, "Scripts", "PlotFunctions.R"))
source(file.path(base_path, "Scripts", "1classSVMFunctions.R"))
source(file.path(base_path, "Scripts", "FeatureGeneration.R"))
#source(file.path(base_path, "Scripts", "DataExploration.R"))

# load in the data and reformat
name <- "Vehkaoja_2018"
data <- fread(file.path(base_path, "Data/Annotated_Vehkaoja_2018.csv"))
data_formatted <- data %>%
  rename(Accelerometer.X = ANeck_x,
         Accelerometer.Y = ANeck_y,
         Accelerometer.Z = ANeck_z,
         ID = DogID,
         Timestamp = t_sec,
         Activity = Behavior_2) %>%
  filter(!Activity == "<undefined>") %>%
  select(ID, Timestamp, Activity, Accelerometer.X, Accelerometer.Y, Accelerometer.Z)


# Split Data ####
# randomly allocate each individual to training, validating, or testing datasets
# pull 10% individuals for the test set
#data_test <- data_formatted[data_formatted$ID %in% sample(unique(data_formatted$ID), ceiling(length(unique(data_formatted$ID)) * 0.1)), ]
#data_other <- anti_join(data_formatted, data_test) # remainder
# save these
#fwrite(data_test, file.path(base_path, "Data/Hold_out_test", paste0(name, "_test.csv")))
#fwrite(data_other, file.path(base_path, "Data/Hold_out_test", paste0(name, "_other.csv")))

# load in 
data_test <- fread(file.path(base_path, "Data/Hold_out_test", paste0(name, "_test.csv")))
data_other <-fread(file.path(base_path, "Data/Hold_out_test", paste0(name, "_other.csv")))

# Visualising data ####
# plot shows sample of each behaviour for each individual
# select a subset of individuals to use
data_subset <- data_formatted[data_formatted$ID %in% unique(data_formatted$ID)[1:5], ]
beh_trace_plot <- plot_behaviours(behaviours = unique(data_subset$Activity), data = data_subset, n_samples = 200, n_col = 4)
beh_volume_plot <- explore_data(data = data_formatted, frequency = 100, colours = unique(data_formatted$ID))


# regroup some of the stationary behaviours together
data_other <- data_other %>%
  mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                           "Stationary", 
                           Activity))

# 1class-SVM model design tuning ####
# list variables to test
targetActivity_options <- c("Galloping", "Stationary", "Walking") #, "Walking", "Panting", "Sitting", "Eating")
down_Hz <- 100
window_length_options <- c(1, 2, 5)
overlap_percent_options <- c(0, 10)
feature_normalisation_options <- c("Standardisation") #, "MinMaxScaling")
nu_options <- c(0.01, 0.1, 0.25, 0.5)
kernel_options <- c("radial", "sigmoid", "polynomial", "linear")
gamma_options <- c(0.001, 0.01)
degree_options <- c(2, 3, 4)

model_hyperparameters_list <- list(
  radial = list(
    gamma = gamma_options
  ),
  polynomial = list(
    gamma = gamma_options,
    degree = degree_options
  ),
  sigmoid = list(
    gamma = gamma_options
  )
)

features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "auto", "zero", "fft")
#features_list <- c("auto", "entropy", "max", "min", "maxVDBA", "sd")
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

# from here on, it will loop
optimal_model_designs <- data.frame()

## Tuning ####
for (targetActivity in targetActivity_options){
  #targetActivity <- "Galloping"
  
  # generate all possible combinations
  options_df <- expand.grid(targetActivity, window_length_options, overlap_percent_options, down_Hz, 
                            feature_normalisation_options, nu_options, kernel_options)
  colnames(options_df) <- c("targetActivity", "window_length", "overlap_percent", "down_Hz", 
                            "feature_normalisation", "nu", "kernel")
  
  # add the additional parameters
  extended_options_df <- create_extended_options(model_hyperparameters_list, options_df)
  
  # create training and validation datasets
  validation_data <- data_other[data_other$ID %in% 
                  sample(unique(data_other$ID), ceiling(length(unique(data_other$ID)) * 0.1)), ]
  training_data <- anti_join(data_other, validation_data) %>%
    filter(Activity == targetActivity)
  
  print("datasets created")
  
  model_tuning_metrics <- model_tuning(extended_options_df, base_path, training_data, validation_data, targetActivity)
  
  print(paste("Model tuning for", targetActivity, "complete"))
  
  # write out the tuning csv
  fwrite(model_tuning_metrics, file.path(base_path, paste(targetActivity, "tuning_metrics.csv", sep = "_")))
}








# Test optimal model ####
# upload csv with the best model designs
optimal_df <- fread(file.path(base_path, "Optimal_Model_Design.csv"))

optimal_model_tests <- data.frame()

targetActivity_options <- c("Galloping")
features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "auto", "zero", "fft")
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

for (activity in targetActivity_options){
  
  # Extract the training and test data
  data_test <- fread(file.path(base_path, "Data", "Hold_out_test", "Vehkaoja_2018_test.csv"))%>%
    mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                             "Stationary",  Activity))
  
  data_other <- fread(file.path(base_path, "Data", "Hold_out_test", "Vehkaoja_2018_other.csv")) %>%
    mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                             "Stationary",  Activity))
  
  evaluation_data <- data_test # generated earlier
  training_data <- data_other %>%
    filter(Activity == activity) %>% 
    na.omit()
  
  # Extract the optimal parameters
  optimal_df_row <- optimal_df %>% as.data.frame() %>% 
    filter(targetActivity == activity) %>%
    mutate(degree = NA) # because I didn't have this lol 
  
  model_evaluation_metrics <- model_testing(optimal_df_row, base_path, training_data, evaluation_data, activity)
  
  print(paste("Optimal model testing for", activity, "complete"))
  
  optimal_model_tests <- rbind(optimal_model_tests, model_evaluation_metrics)
  
}

fwrite(optimal_model_tests, file.path(base_path, "Optimal_Model_Test_Balanced_2.csv"))








# stitch together ####
locomotion <- fread(file.path(base_path, "Bubbles_Locomotion_predictions.csv")) %>%
  rename(Locomotion_reference = reference,
         Locomotion_predicted = predicted) %>%
  mutate(
    Locomotion_reference = ifelse(Locomotion_reference == "Normal", "Locomotion", Locomotion_reference),
    Locomotion_predicted = ifelse(Locomotion_predicted == "Normal", "Locomotion", Locomotion_predicted)
  )


inactive <- fread(file.path(base_path, "Bubbles_Inactive_predictions.csv"))
# inactive is 2 seconds whereas locomotion is 1 second 
inactive_expanded <- inactive %>%
  slice(rep(1:n(), each = 2)) %>%  # Duplicate each row
  mutate(across(everything(), ~replace(., row_number() %% 2 == 0, NA)))  # Replace every second row with NA
inactive_expanded <- inactive_expanded[1:length(inactive_expanded$Timestamp)-1] %>%
  rename(Inactive_reference = reference,
         Inactive_predicted = predicted) %>%
  select(Inactive_reference, Inactive_predicted) %>%
  mutate(
    Inactive_reference = ifelse(Inactive_reference == "Normal", "Inactive", Inactive_reference),
    Inactive_predicted = ifelse(Inactive_predicted == "Normal", "Inactive", Inactive_predicted)
  )

combine <- cbind(inactive_expanded, locomotion) %>%
  select(ID, Timestamp, Inactive_reference, Inactive_predicted, Locomotion_reference, Locomotion_predicted)

combine_agg <- combine %>%
  na.omit() %>%
  mutate(
    Predicted = ifelse(
      Inactive_predicted == "Inactive" & Locomotion_predicted == "Outlier", "Inactive",
      ifelse(
        Inactive_predicted == "Outlier" & Locomotion_predicted == "Locomotion", "Locomotion",
        ifelse(
          Inactive_predicted == "Outlier" & Locomotion_predicted == "Outlier", "Outlier",
          "disagreement"
        )
      )
    ),
    Actual = ifelse(
      Inactive_reference == "Inactive" & Locomotion_reference == "Outlier", "Inactive",
      ifelse(
        Inactive_reference == "Outlier" & Locomotion_reference == "Locomotion", "Locomotion",
        ifelse(
          Inactive_reference == "Outlier" & Locomotion_reference == "Outlier", "Outlier",
          "disagreement"
        )
      )
    )
  ) %>%
  select(ID, Timestamp, Predicted, Actual)

summary <- combine_agg %>%
  group_by(Predicted, Actual) %>%
  count()

# plot 
custom_colors <- c("Locomotion" = "#6A4C93", "Inactive" = "#F3B61F", "Outlier" = "#D7263D")

ggplot(combine_agg, aes(x = Predicted, fill = Actual)) +
  geom_bar(position = "stack") +
  scale_fill_manual(values = custom_colors) +
  labs(x = "Predicted Class", y = "Count", fill = "Actual Class") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
