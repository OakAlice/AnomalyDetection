# Initialisation Script

# Set up ####
# load packages
library(data.table)
library(tidyverse)
library(future)
library(future.apply)
library(e1071)
library(h2o)


# set base path
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"

# load in the files
source(file.path(base_path, "Scripts", "PlotFunctions.R"))
source(file.path(base_path, "Scripts", "Functions.R"))
source(file.path(base_path, "Scripts", "FeatureGeneration.R"))

# load in the data
data_original <- fread(file.path(base_path, "Data", "Annotated_Jeantet_2020.csv"))
# Split Data ####
# randomly allocate each individual to training, validating, or testing datasets
all_individuals <- sample(unique(data_original$ID))
data_test <- data_original[data_original$ID %in% all_individuals[1:4], ]
other_data <- anti_join(data_original, data_test)


# Visualising data ####
# plot shows sample of each behaviour for each individual
# select a subset of individuals to use
data_subset <- data_selected[data_selected$ID %in% unique(data_selected$ID)[1:5], ]
beh_trace_plot <- plot_behaviours(behaviours = unique(data_subset$Activity), data = data_subset, n_samples = 200, n_col = 4)
# plot shows the total samples per behaviour and individual
beh_volume_plot <- explore_data(data = data_selected, frequency = 25, colours = unique(data_selected$ID))


# Anomaly Detection Analysis ####

# 1class-SVM ####
# list variables to test
targetActivity_options <- c("breathing", "staying_at_surface", "resting")
window_length_options <- c(1, 3, 5)
overlap_percent_options <- c(0)
freq_Hz <- 20
feature_normalisation_options <- c("Standardisation") # "MinMaxScaling"
nu_options <- c(0.4, 0.5, 0.6, 0.7)
kernel_options <- c("radial", "polynomial") 
features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "auto") # zero
validation_individuals <- 2

# from here on, it will loop
optimal_model_designs <- data.frame()

for (targetActivity in targetActivity_options){
  # Tuning ##
  # generate all possible combinations
  options_df <- expand.grid(targetActivity, architecture_options, window_length_options, overlap_percent_options, freq_Hz, 
                            feature_normalisation_options, nu_options, kernel_options)
  colnames(options_df) <- c("targetActivity", "architecture", "window_length", "overlap_percent", "frequency_Hz", 
                            "feature_normalisation", "nu", "kernel")
  
  # randomly create training and validation datasets
  datasets <- create_datasets(data = other_data, targetActivity, validation_individuals)
  data_training <- datasets$data_training %>% select(-time, -ID)
  data_validation <- datasets$data_validation %>% select(-time, -ID)
  print("datasets created")
  
  model_tuning_metrics <- model_tuning(options_df, base_path, data_training, data_validation, targetActivity)
  
  print(paste("Model tuning for", targetActivity, "complete"))
  
  # write out the tuning csv
  fwrite(model_tuning_metrics, file.path(base_path, paste(targetActivity, "tuning_metrics.csv", sep = "_")))
}


# Autoencoder ####
# specify the training data
targetActivity <- "breathing"
validation_individuals <- 2

# extract the training and validation sets
training_data <- other_data %>% filter(ID %in% unique(other_data$ID)[1:(length(unique(other_data$ID)) - validation_individuals)])
validation_data <- anti_join(other_data, training_data)

# split the training data into target and anomalous behaviours
target_training_data <- training_data %>% filter(Activity == targetActivity) %>% 
  select(-Activity, -time, -ID) %>% mutate(Label = 0)
anom_training_data <- training_data %>% filter(!Activity == targetActivity) %>%
  select(-Activity, -time, -ID) %>% mutate(Label = 1)

# train the detector on normal data only
h2o.init()

# convert data to h2o objects
train_h2o = as.h2o(target_training_data)
test_h2o = as.h2o(anom_training_data)

# build auto encoder model with 3 layers
model_unsup = h2o.deeplearning(x = 1:3,
                               training_frame = train_h2o,
                               model_id = "Test01",
                               autoencoder = TRUE,
                               reproducible = TRUE,
                               ignore_const_cols = FALSE,
                               seed = 42,
                               hidden = c(50,10,50,100,100),
                               epochs = 100,
                               activation ="Tanh")
# view the model
model_unsup

# now we need to calculate MSE or anomaly score  
anmlt = h2o.anomaly(model_unsup, 
                    train_h2o, 
                    per_feature = FALSE) %>% as.data.frame()
# create a label for healthy data
anmlt$y = 0
# view top data
head(anmlt)


# calculate thresholds from train data
threshold = quantile(anmlt$Reconstruction.MSE, probs = 0.999)
