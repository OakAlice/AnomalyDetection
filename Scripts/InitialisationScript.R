# Initialisation Script

# Set up ####
# load packages
library(pacman)
p_load(data.table, tidyverse, future.apply, e1071, zoo, 
       tsfeatures, umap, plotly, randomForest)
library(h2o)

# set base path
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"

# load in the scripts
scripts <- list("Dictionaries.R", "PlotFunctions.R", "FeatureGeneration.R") #, "DataExploration.R")

for (script in scripts){
  source(file.path(base_path, "Scripts", script))
}

# Dataset selection ####
dataset_name <- "Vehkaoja_Dog"
list_name <- all_dictionaries[[dataset_name]]
movement_data <- get(list_name)

# Explore dataset ####
# load in data
move_data <- fread(file.path(base_path, "Data", paste0(movement_data$name, "_Corrected.csv")))

# generate plots
# volume of data per individual and activity
plot_activity_ID_graph <- plot_activity_ID(data = move_data, 
                                frequency = movement_data$Frequency, colours = length(unique(move_data$ID)))
# examples of each trace
plot_trace_example_graph <- plot_trace_example(behaviours = unique(move_data$Activity), 
          data = move_data, n_samples = 200, n_col = 4)

# other exploratory plots

# Split Data ####
# randomly allocate each individual to training, validating, or testing datasets
# pull 10% individuals for the test set
  #data_test <- move_data[move_data$ID %in% sample(unique(move_data$ID), ceiling(length(unique(move_data$ID)) * 0.1)), ]
  #data_other <- anti_join(move_data, data_test) # remainder
# save these
  #fwrite(data_test, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
  #fwrite(data_other, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))

# load in 
  data_test <- fread(file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
  data_other <-fread(file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))

# Feature Generation ####
## Prt 1: Determine Window Length ####
  duration_info <- average_duration(data = data_other, sample_rate = movement_data$Frequency)
  stats <- duration_info$duration_stats %>% 
    filter(Activity %in% movement_data$target_behaviours)
  plot <- duration_info$duration_plot

# therefore, select the window length and add that to the Dictionary  
  
## Prt 2: Determine Features ####
### generate a tonne of features ####
  all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
  
  # window data and generate features
  example_data <- data_other %>% slice(1:100000)
  feature_data <- generate_features(movement_data, data = example_data, normalise = "z_scale")
 # fwrite(feature_data, file.path(base_path, "Output", "DogFeatureData_1000.csv"))

### select the best features ####
#### UMAP ####
# UMAP is a non-linear version of PCA, we use this to reduce dimensionality
  
# define the hyperparameters
minimum_distance <- 0.7
num_neighbours <- 10
shape_metric <- 'manhattan' # 'euclidean'
  
UMAP_representations <- UMAP_reduction(feature_data, minimum_distance, num_neighbours, shape_metric)
UMAP_representations$UMAP_2D_plot
UMAP_representations$UMAP_3D_plot
UMAP_features <- ###
  
#### PCA ####
pca_results <- PCA_reduction(feature_data)
pca_result <- pca_results$pca_result
pca_df <- pca_results$pca_df
scree_plot(pca_result)
scatter_plot_pca(pca_df)
PCA_features <- pca_df[, c(1:10, which(names(pca_df) == "Activity"))]

#### LDA ####
number_features <- 30
lda_results <- LDA_feature_selection(feature_data, target_column = "Activity", number_features)
print(lda_results$Feature_Importance_Plot)
LDA_features <- lda_results$Selected_Features[1:20, 1]

#### Random Forest ####
num_trees <- 50
number_features <- 50
rf_results <- RF_feature_selection(feature_data, target_column = "Activity", n_trees = num_trees, number_features = 100)
print(rf_results$Feature_Importance_Plot)
print(rf_results$OOB_Error_Plot)
rf_features <- rf_results$Selected_Features[1:100, ]

# these possible optimal feature sets will then be explored in the model tuning
# Tuning various classifiers ####  

classifier_types <- c("SingleClassSVM")




  
# list variables to test
targetActivity_options <- c("Galloping", "Stationary", "Walking") #, "Walking", "Panting", "Sitting", "Eating")
down_Hz <- 100
window_length_options <- c(1, 2, 5)
overlap_percent_options <- c(0, 10)
feature_sets <- c(UMAP_features, PCA_features, LDA_features, rf_features)
feature_normalisation_options <- c("Standardisation") #, "MinMaxScaling")

# SVM options
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
