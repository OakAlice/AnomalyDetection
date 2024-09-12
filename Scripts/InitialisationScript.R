# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Set Up
# ---------------------------------------------------------------------------

# load packages
library(pacman)
p_load(data.table, tidyverse, future.apply, e1071, zoo, caret,
       tsfeatures, umap, plotly, randomForest, pROC, purrr)
#library(h2o) # for UMAP, but takes a while so ignore unless necessary

# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
#setwd("C:/Users/oaw001/Documents/AnomalyDetection")

# load in the scripts
scripts <- list("Dictionaries.R", "PlotFunctions.R", "FeatureGeneration.R",
                "FeatureSelection.R", "OtherFunctions.R",
                "UserInput.R") #, "DataExploration.R")

successful <- TRUE
for (script in scripts) {
  tryCatch(source(file.path(base_path, "Scripts", script)), 
           error = function(e) successful <<- FALSE)
}

print(if (successful) "All scripts sourced successfully!" else "Error sourcing one or more scripts.")

# ---------------------------------------------------------------------------
# Define parameters and create data splits for this particular run
# ---------------------------------------------------------------------------

dataset_name <- "Vehkaoja_Dog"
list_name <- all_dictionaries[[dataset_name]]
movement_data <- get(list_name)

# load in data
move_data <- fread(file.path(base_path, "Data", paste0(movement_data$name, "_Corrected.csv")))

# Split Data ####
if (file.exists(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_Labelled_test.csv")))){
  # if this has been run before, just load in the split data  
    data_test <- fread(file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
    data_other <-fread(file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))
  } else {
    # if this is the first time running code for this dataset, create hold-out test set
    data_test <- move_data[move_data$ID %in% sample(unique(move_data$ID), ceiling(length(unique(move_data$ID)) * test_proportion)), ]
    data_other <- anti_join(move_data, data_test) # remainder
  # save these
    fwrite(data_test, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
    fwrite(data_other, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))
  }

# explore data in PreProcessingDecisions.R to determine window length and behavioural clustering
# when complete, add to dictionary

# ---------------------------------------------------------------------------
# Feature Generation and Elimination
# ---------------------------------------------------------------------------
# generate all features, unless that has already been done 
#(it can be very time consuming so I tend to save it when I've done it)

if (file.exists(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))){
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
} else {
  feature_data <- generate_features(movement_data, data = data_other, 
                                    normalise = "z_scale", features = features_type)
}

# eliminate features that don't contribute information
potential_features <- select_potential_features(feature_data, threshold = 0.9)
subset_data <- feature_data %>% select(c(all_of(potential_features), Activity, Time, ID))

# ---------------------------------------------------------------------------
# Tuning model hyperparameters
# ---------------------------------------------------------------------------
# this section of the code iterates through hyperparameter combinations for OCC
# AUC for the target class is recorded and saved in the output table

# create all the options
extended_options_df <- create_extended_options(model_hyperparameters_list, options_df)
extended_options_df2 <- create_extended_options2(feature_hyperparameters_list, extended_options_df)

model_outcome <- data.frame() # for each model

for (i in 1:nrow(extended_options_df)){
  
  # select row
  options <- extended_options_df2[i, ] %>% data.frame()
  
  cross_outcome <- data.frame() # for each cross-validation fold
  
  # begin cross validation
 for (k in 1:k_folds){ 
  
   # create training and validation data
   validation_data <- subset_data[subset_data$ID %in% sample(unique(subset_data$ID), ceiling(length(unique(subset_data$ID)) * validation_proportion)), ]
   suppressMessages({
       training_data <- anti_join(subset_data, validation_data)
   })
   
  #### feature selection ####
     if (feature_selection == "UMAP") {  # TODO: increase number of embeddings
       
       training_labels <- training_data %>% na.omit() %>% select(Activity)
       training_numeric <- training_data %>% na.omit() %>% select(-Activity, -Time, -ID)
       
       UMAP_representations <- UMAP_reduction(numeric_features= training_numeric, 
                                              labels = training_labels, 
                                              minimum_distance = options$min_dist, 
                                              num_neighbours = options$n_neighbours, 
                                              shape_metric = options$metric,
                                              save_model_path = file.path(base_path, "Output"))
       #UMAP_representations$UMAP_2D_plot
       #UMAP_representations$UMAP_3D_plot
       selected_feature_model <- UMAP_representations$UMAP_2D_model
       
       # just get data from the training class to train the OCC
       selected_feature_data <- as.data.frame(UMAP_representations$UMAP_2D_embeddings)
       
     } else if (feature_selection == "RF"){
       RF_features <- RF_feature_selection(training_data, 
                                           target_column = "Activity", 
                                           n_trees = options$number_trees, 
                                           number_features = options$number_features)
       
       top_features <- RF_features$Selected_Features$Feature[1:number_features]
       #RF_features$Feature_Importance_Plot
       #RF_features$OOB_Error_Plot
       selected_feature_data <- training_data %>% select(c(all_of(top_features), Activity, Time, ID))
     }
   
  #### train model ####
       #add in the other types later
    if (options$model == "SVM"){
      params <- list(
        gamma = options$gamma,
        degree = options$degree
      )
      params <- Filter(Negate(is.na), params)
      
      # select just the target class data for the OCC training
      target_class_feature_data <- selected_feature_data %>% filter(Activity == as.character(options$targetActivity)[1]) %>% 
        select(-Activity, -Time, -ID)
      
      single_class_SVM <- do.call(svm, c(
        list(
          target_class_feature_data, 
          y = NULL, 
          type = 'one-classification',
          nu = options$nu,
          scale = TRUE,
          kernel = options$kernel
        ),
        params  # Add other parameters
      ))
    }
    
  #### validate model ####
    # convert validation data to the same shape and features as the training data
    if(feature_selection == "UMAP") {
      
      validation_numeric <- validation_data %>% select(-Activity, -Time, -ID)
      umap_model <- readRDS(file.path(base_path, "Output", "umap_2D_model.rds")) # read it in
      selected_validation_data <- predict(umap_model, validation_numeric) %>% as.data.frame
      colnames(selected_validation_data) <- c("UMAP1", "UMAP2")
      
    } else if (feature_selection == "RF"){
      selected_validation_data <- validation_data %>% select(c(all_of(top_features))) %>% na.omit()
    }
   
   # make predictions
    ground_truth_labels <- validation_data %>% na.omit() %>% select(Activity)
    ground_truth_labels <- ifelse(ground_truth_labels == as.character(options$targetActivity), 1, -1)   
    decision_scores <- predict(single_class_SVM, newdata = selected_validation_data, decision.values = TRUE)
    scores <- as.numeric(attr(decision_scores, "decision.values"))
    
    # calculate AUC
    roc_curve <- roc(as.vector(ground_truth_labels), scores)
    auc_value <- auc(roc_curve)
    
    # Plot the ROC curve if you want to see it
    # plot(roc_curve, col = "#1c61b6", main = paste("ROC Curve (AUC =", round(auc_value, 4), ")"))
    
    # save a row of parameters and values
    cross_result <- cbind(Model = as.character(options[1, "model"]),
                          Activity = as.character(options[1, "targetActivity"]), 
                          nu = as.character(options[1, "nu"]), 
                          gamma = as.character(options[1, "gamma"]), 
                          kernel = as.character(options[1, "kernel"]),
                          min_dist = as.character(options[1, "min_dist"]),
                          n_neighbours = as.character(options[1, "n_neighbours"]),
                          metric = as.character(options[1, "metric"]),
                          number_features = as.character(options[1, "number_features"]),
                          number_trees = as.character(options[1, "number_trees"]),
                          feature_method = as.character(options[1, "feature_selection"]),
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

# save to
model_outcome <- as.data.frame(model_outcome)
ensure.dir(file.path(base_path, "Output", dataset_name))
fwrite(model_outcome, file.path(base_path, "Output", dataset_name, paste0(dataset_name, "_model_outcomes.csv")))

# ---------------------------------------------------------------------------
# Testing highest performing model hyperparameters
# ---------------------------------------------------------------------------

# in progress
