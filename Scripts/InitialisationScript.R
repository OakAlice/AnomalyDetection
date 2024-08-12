# Initialisation Script

# Set up ####
# load packages
library(data.table)
library(tidyverse)
library(future)
library(future.apply)
library(e1071)

# set base path
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"

# load in the files
source(file.path(base_path, "Scripts", "PlotFunctions.R"))
source(file.path(base_path, "Scripts", "Functions.R"))
source(file.path(base_path, "Scripts", "FeatureGeneration.R"))

# load in the data and select relevant columns
data_original <- fread(file.path(base_path, "Data", "Annotated_Jeantet_2020.csv"))
data_selected <- data_original %>%
  select(LCOLOUR, RCOLOUR, TIME, BEHAV, X, Y, Z, temp) %>%
  mutate(ID = paste0(LCOLOUR, RCOLOUR)) %>% # create a unique identifier
  rename(Time = TIME, # rename columns to preference
         Activity = BEHAV,
         Accelerometer.X = X,
         Accelerometer.Y = Y,
         Accelerometer.Z = Z,
         Temperature = temp) %>%
  select(-c(LCOLOUR, RCOLOUR)) %>%
  mutate(ID = as.numeric(factor(ID))) # convert to numeric for neatness

# Visualising data ####
# plot shows sample of each behaviour for each individual
# select a subset of individuals to use

data_selected <- data_original
data_subset <- data_selected[data_selected$ID %in% unique(data_selected$ID)[1:5], ]
beh_trace_plot <- plot_behaviours(behaviours = unique(data_subset$Activity), data = data_subset, n_samples = 200, n_col = 4)
# plot shows the total samples per behaviour and individual
beh_volume_plot <- explore_data(data = data_selected, frequency = 25, colours = unique(data_selected$ID))

# Group Behaviours ####
#data_selected <- data_selected %>%
#  na.omit() %>%
#  mutate(Activity = ifelse(Activity == "notMoving" | Activity == "Nest", "Stationary", Activity))

# Split Data ####
# randomly allocate each individual to training, validating, or testing datasets
all_individuals <- sample(unique(data_selected$ID))
data_test <- data_selected[data_selected$ID %in% all_individuals[1:5], ]

# balance the occurances of each behaviour in the validation and testing datasets
# do this to make the performance metrics more meaningful

# Anomaly Detection Analysis ####

# group behaviours
#data_ad <- data_selected %>%
#  na.omit() %>%
#  mutate(Activity = ifelse(Activity == "notMoving" | Activity == "Nest", "Stationary", Activity))

data_ad <- data_selected
other_data <- anti_join(data_ad, data_test)

# list variables to test
targetActivity_options <- c("resting", "swimming", "breathing", "staying_at_surface")
window_length_options <- c(1, 2, 5, 10, 15, 20)
overlap_percent_options <- c(0, 50)
freq_Hz <- 20
feature_normalisation_options <- c("MinMaxScaling", "Standardisation")
nu_options <- c(0.001, 0.01, 0.05, 0.1, 0.05, 0.2, 0.5)
kernel_options <- c("radial", "linear", "hyperbolic", "sigmoid", "polynomial", "RBF") 
features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "zero", "auto")
validation_individuals <- 10

# from here on, it will loop
optimal_model_designs <- data.frame()

for (targetActivity in targetActivity_options){
  # Tuning ####
  # generate all possible combinations
  options_df <- expand.grid(targetActivity, window_length_options, overlap_percent_options, freq_Hz, 
                            feature_normalisation_options, nu_options, kernel_options)
  colnames(options_df) <- c("targetActivity", "window_length", "overlap_percent", "frequency_Hz", 
                            "feature_normalisation", "nu", "kernel")
  
  # randomly create training and validation datasets
  datasets <- create_datasets(other_data, targetActivity, validation_individuals)
  data_training <- datasets$data_training %>% select(-time, -ID)
  data_validation <- datasets$data_validation %>% select(-time, -ID)
  print("datasets created")
  
  model_tuning_metrics <- model_tuning(options_df, base_path, data_training, data_validation, targetActivity)
  
  print(paste("Model tuning for", targetActivity, "complete"))
  
  # write out the tuning csv
  fwrite(model_tuning_metrics, file.path(base_path, paste(targetActivity, "tuning_metrics.csv", sep = "_")))
}
