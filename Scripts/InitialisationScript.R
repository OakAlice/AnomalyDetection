# Initialisation Script

# Set up ####
# load packages
library(pacman)
p_load(data.table, tidyverse, future.apply, e1071, zoo, 
       tsfeatures, umap, plotly, randomForest, pROC)
#library(h2o)

# set base path
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"

# load in the scripts
scripts <- list("Dictionaries.R", "PlotFunctions.R", "FeatureGeneration.R",
                "FeatureSelection.R", "OtherFunctions.R") #, "DataExploration.R")

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
  
## Prt 2: Preprocess and Generate Features ####
### generate a tonne of features ####
  all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
  
  # window data and generate features
  suppressMessages({
    example_data <- data_other %>% group_by(ID, Activity) %>% 
      filter(ID %in% unique(data_other$ID)[1:5]) %>% slice(1:1000)
  })
  suppressMessages({
    feature_data <- generate_features(movement_data, data = example_data, normalise = "z_scale")
  })
 # fwrite(feature_data, file.path(base_path, "Output", "DogFeatureData_1.csv"))
 # feature_data <- fread(file.path(base_path, "Output", "DogFeatureData_1000.csv"))

### insert other preprocessing steps later ####  
  
# Tuning models ####
## Defining parameters ####
targetActivity_options <- c("Walking")
classifier_types <- c("SVM" ) #, "Autoencoder", "GMM", "PPNN", "KNN")
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

### tunable feature parameters ####
#### UMAP ####
minimum_distance_options <- c(0.7)
num_neighbours_options <- 10
shape_metric_options <- 'manhattan' # 'euclidean'

feature_sets <- c("UMAP")
feature_normalisation_options <- c("z-scale") # "Standardisation", "MinMaxScaling"

feature_hyperparameters_list <- list(
  UMAP = list(min_dist = minimum_distance_options,
              n_neighbours = num_neighbours_options,
              metric = shape_metric_options)
)

### tunable model parameters ####
### SVM specific options ####
nu_options <- c(0.01)
kernel_options <- c("radial" ) #, "sigmoid", "polynomial", "linear")
gamma_options <- c(0.001)
degree_options <- c(3)

model_hyperparameters_list <- list(
  radial = list(gamma = gamma_options),
  polynomial = list(gamma = gamma_options, degree = degree_options),
  sigmoid = list(gamma = gamma_options))

### All possible parameter sets ####
# make options_df for all of these options # TODO: Is just for SVM right now
options_df <- expand.grid(targetActivity_options, 
                          feature_sets, 
                          feature_normalisation_options, 
                          nu_options, 
                          kernel_options)
colnames(options_df) <- c("targetActivity", "feature_sets", "feature_normalisation", "nu", "kernel")

# add the additional parameters
extended_options_df <- create_extended_options(model_hyperparameters_list, options_df)
extended_options_df2 <- create_extended_options2(feature_hyperparameters_list, extended_options_df)
  
# Tune Models ####
model_outcome <- data.frame() # for each model

for (i in 1:nrow(extended_options_df)){
  
  # select row
  options <- extended_options_df2[i, ] %>% data.frame()
  
  cross_outcome <- data.frame() # for each cross-validation fold
  
  # begin cross validation
 for (k in 1:k_folds){ 
  
   # create training and validation data
   validation_data <- feature_data[feature_data$ID %in% sample(unique(feature_data$ID), ceiling(length(unique(feature_data$ID)) * 0.2)), ]
   suppressMessages({
       training_data <- anti_join(feature_data, validation_data)
       # TODO: This line needs to be moved down
   })
   
  # extract numeric elements from feature_data # TODO: add automated removal of NA columns
   selected_training_data <- training_data %>%
     select(-Time, -ID, -Y_zero_proportion, -X_nperiods, -X_seasonal_period, -Y_nperiods, -Y_seasonal_period, -Z_nperiods, -Z_seasonal_period, -X_alpha, -X_beta, -X_gamma, -Y_alpha, -Y_beta, -Y_gamma, -Z_alpha, -Z_beta, -Z_gamma) %>% # these were all NA
     na.omit()
   training_labels <- selected_training_data %>% select(Activity)
   training_numeric <- selected_training_data %>% select(-Activity)
   
  #### select the feature subset ####
     if (options$feature_sets == "UMAP") {  # TODO: increase number of embeddings
       UMAP_representations <- UMAP_reduction(training_numeric, training_labels, 
                                              minimum_distance = options$min_dist, 
                                              num_neighbours = options$n_neighbours, 
                                              shape_metric = options$metric,
                                              save_model_path = file.path(base_path, "Output"))
       #UMAP_representations$UMAP_2D_plot
       #UMAP_representations$UMAP_3D_plot
       selected_feature_model <- UMAP_representations$UMAP_2D_model
       # just get data from the training class to train the OCC
       selected_feature_data <- as.data.frame(UMAP_representations$UMAP_2D_embeddings) %>%
         filter(Activity == as.character(options$targetActivity)[1]) %>%
         select(-Activity)
       
     # Todo: add LDA, PCA, and RF
    
  #### train model ####
       #add in the other types later
    if (model == "SVM"){
      params <- list(
        gamma = options$gamma,
        degree = options$degree
      )
      params <- Filter(Negate(is.na), params)
      
      single_class_SVM <- do.call(svm, c(
        list(
          selected_feature_data, 
          y = NULL,  # No response variable for one-class SVM
          type = 'one-classification',
          nu = options$nu,
          scale = TRUE,
          kernel = options$kernel
        ),
        params  # Add filtered parameters
      ))
    }
    
    #### validate model ####
    # make into the same shape as the other data
       selected_validation_data <- training_data %>%
         select(-Time, -ID, -Y_zero_proportion, -X_nperiods, -X_seasonal_period, -Y_nperiods, -Y_seasonal_period, -Z_nperiods, -Z_seasonal_period, -X_alpha, -X_beta, -X_gamma, -Y_alpha, -Y_beta, -Y_gamma, -Z_alpha, -Z_beta, -Z_gamma) %>% # these were all NA
         na.omit()
       validation_labels <- selected_validation_data %>% select(Activity)
       numeric_validation_data <- selected_validation_data %>% select(-Activity)
       
    if(options$feature_sets == "UMAP") {
      umap_model <- readRDS(file.path(base_path, "Output", "umap_2D_model.rds"))
      transformed_data <- predict(umap_model, numeric_validation_data) %>%
        as.data.frame
      colnames(transformed_data) <- c("UMAP1", "UMAP2")
    } 
      
    # use the previously made SVM to predict on this new data
    decision_scores <- predict(single_class_SVM, newdata = transformed_data, decision.values = TRUE)
    scores <- as.numeric(attr(decision_scores, "decision.values"))
    
    # get the real labels
    ground_truth_labels <- ifelse(validation_labels$Activity == options$targetActivity, 1, -1)   
    
    # calculate AUC
    roc_curve <- roc(ground_truth_labels, scores)
    auc_value <- auc(roc_curve)
    
    # Plot the ROC curve
    #AUC_ROC_Curve <- plot(roc_curve, col = "#1c61b6", main = paste("ROC Curve (AUC =", round(auc_value, 4), ")"))
    
    # save a row of parameters and values
    cross_result <- cbind(Model = "SVM",
                          Activity = as.character(options[1, "targetActivity"]), 
                          nu = as.character(options[1, "nu"]), 
                          gamma = as.character(options[1, "gamma"]), 
                          kernel = as.character(options[1, "kernel"]),
                          feature_method = as.character(options[1, "feature_sets"]),
                          min_dist = as.character(options[1, "min_dist"]),
                          n_neighbours = as.character(options[1, "n_neighbours"]),
                          metric = as.character(options[1, "metric"]),
                          feature_normalisation = as.character(options[1, "feature_normalisation"]),
                          AUC_Value = as.numeric(auc_value)
                          ) %>% data.frame()
    
    # save to the loop
    cross_outcome <- rbind(cross_outcome, cross_result)
  }
  
  # take the mean and standard deviations of the three cross-validation AUC results
   suppressMessages({
     cross_average <- cross_outcome %>%
       # Exclude 'AUC_Value' from the grouping columns
       group_by(across(-AUC_Value)) %>%  
       mutate(AUC_Value = as.numeric(AUC_Value)) %>%
       summarise(
         mean_AUC = mean(AUC_Value, na.rm = TRUE),  # Calculate the mean AUC
         sd_AUC = sd(AUC_Value, na.rm = TRUE)       # Calculate the standard deviation of AUC
       )
   })
  
  # save to the dataframe.
  model_outcome <- rbind(model_outcome, cross_average)
 }
}





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
