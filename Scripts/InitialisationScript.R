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
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
# and working directory so h2o works # can't be on the server
setwd("C:/Users/oaw001/Documents/AnomalyDetection")

# load in the files
source(file.path(base_path, "Scripts", "PlotFunctions.R"))
source(file.path(base_path, "Scripts", "1classSVMFunctions.R"))
source(file.path(base_path, "Scripts", "FeatureGeneration.R"))
source(file.path(base_path, "Scripts", "AutoEncoderFunctions.R"))

# load in the data
data_original <- fread(file.path(base_path, "Data", "Annotated_Jeantet_2020.csv"))

# jordan data
data_original <- fread("C:/Users/oaw001/Documents/Perentie/DiCicco_Perentie_Labelled.csv")
data_original <- data_original %>%
  rename(Accelerometer.X = Accel_X,
         Accelerometer.Y = Accel_Y,
         Accelerometer.Z = Accel_Z) %>%
  filter(!Activity == "NaN")
base_path <- "C:/Users/oaw001/Documents/Perentie"


# Split Data ####
# randomly allocate each individual to training, validating, or testing datasets
all_individuals <- sample(unique(data_original$ID))
data_test <- data_original[data_original$ID %in% all_individuals[1], ]
other_data <- anti_join(data_original, data_test)


# Visualising data ####
# plot shows sample of each behaviour for each individual
# select a subset of individuals to use
data_subset <- data_original[data_original$ID %in% unique(data_original$ID)[1:3], ]
beh_trace_plot <- plot_behaviours(behaviours = unique(data_subset$Activity), data = data_subset, n_samples = 200, n_col = 4)
# plot shows the total samples per behaviour and individual
beh_volume_plot <- explore_data(data = data_original, frequency = 25, colours = unique(data_original$ID))


# Anomaly Detection Analysis ####

# 1class-SVM ####
# list variables to test
targetActivity_options <- c("Inactive", "Locomotion")
window_length_options <- c(1)
overlap_percent_options <- c(0)
freq_Hz <- 50
feature_normalisation_options <- c("Standardisation") # "MinMaxScaling"
nu_options <- c(0.4)
kernel_options <- c("radial") 
features_list <- c("mean", "max", "min", "sd", "cor", "SMA", "minODBA", "maxODBA", "minVDBA", "maxVDBA", "entropy", "auto") # zero
validation_individuals <- 1
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
# need to add this into the rest of the code 


# from here on, it will loop
optimal_model_designs <- data.frame()

for (targetActivity in targetActivity_options){
  # Tuning ##
  # generate all possible combinations
  options_df <- expand.grid(targetActivity, window_length_options, overlap_percent_options, freq_Hz, 
                            feature_normalisation_options, nu_options, kernel_options)
  colnames(options_df) <- c("targetActivity", "window_length", "overlap_percent", "frequency_Hz", 
                            "feature_normalisation", "nu", "kernel")
  
  # randomly create training and validation datasets
  datasets <- create_datasets(data = other_data, targetActivity, validation_individuals)
  data_training <- datasets$data_training %>% select(-Timestamp, -ID)
  data_validation <- datasets$data_validation %>% select(-Timestamp, -ID)
  print("datasets created")
  
  model_tuning_metrics <- model_tuning(options_df, base_path, data_training, data_validation, targetActivity)
  
  print(paste("Model tuning for", targetActivity, "complete"))
  
  # write out the tuning csv
  fwrite(model_tuning_metrics, file.path(base_path, paste(targetActivity, "tuning_metrics.csv", sep = "_")))
}




# Autoencoder ####
# specify the training data
training_data <- other_data %>% filter(ID %in% unique(other_data$ID)[1:(length(unique(other_data$ID)) - validation_individuals)])
validation_data <- anti_join(other_data, training_data)

# variables to iterate through
model_name <- "Test01"
targetActivity <- "staying_at_surface"
validation_individuals <- 2
window_size <- 100
hidden_layers <- c(50,10,50,100,100)
epoch_option <- 100
activation_shape <- "Tanh"
threshold_probability <- 0.75

# initialise process
h2o.init()

# create target and anomalous data, formatted and windowed
h2o_objects <- create_format_data(training_data, targetActivity, window_size)
target_windows <- h2o_objects$target_windows
anom_windows <- h2o_objects$anom_windows

# Train the autoencoder as before
model_unsup = h2o.deeplearning(
  x = 1:length(target_windows), # only predictors, no y value (label)
  training_frame = target_windows,
  model_id = model_name,
  autoencoder = TRUE,
  seed = 42,
  hidden = hidden_layers,
  epochs = epoch_option,
  activation = activation_shape
)

# Calculate the reconstruction error for normal data
anmlt = h2o.anomaly(model_unsup, target_windows, per_feature = FALSE) %>% as.data.frame()
anmlt$y = 0
threshold = quantile(anmlt$Reconstruction.MSE, probs = threshold_probability)

# Calculate the reconstruction error for anomaly data
test_anmlt = h2o.anomaly(model_unsup, anom_windows, per_feature = FALSE) %>% as.data.frame()
test_anmlt$y = 1

# Combine the results
results = data.frame(rbind(anmlt, test_anmlt), threshold)
head(results)

# plot
error_distribution <- ggplot(results, aes(x = Reconstruction.MSE, fill = factor(y))) +
  geom_histogram(binwidth = 0.00001, position = "identity", alpha = 0.6) +  # Use a smaller binwidth
  #scale_x_log10() +  # Logarithmic scale to handle small MSE values
  labs(title = "Frequency Distribution of Reconstruction MSE",
       x = "Reconstruction MSE",
       y = "Frequency",
       fill = "Class") +
  theme_minimal()
error_distribution




# understand 
summary <- results %>%
  group_by(y) %>%
  summarise(count = n(),
            min = min(Reconstruction.MSE),
            max = max(Reconstruction.MSE),
            median = median(Reconstruction.MSE))



# Adjust plot sizes
options(repr.plot.width = 15, repr.plot.height = 6)
plot(results$Reconstruction.MSE, type = 'n', xlab='observations', ylab='Reconstruction.MSE', main = "Anomaly Detection Results")
points(results$Reconstruction.MSE, pch=19, col=ifelse(results$Reconstruction.MSE < threshold, "green", "red"))
abline(h=threshold, col='red', lwd=2)






