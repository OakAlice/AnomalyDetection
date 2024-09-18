#---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data                  ####
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Set Up                                                                 ####
#---------------------------------------------------------------------------

# load packages
library(pacman)
p_load(data.table, tidyverse, purrr, future.apply, e1071, zoo, caret,
       tsfeatures, umap, plotly, randomForest, pROC, bench, PRROC)
#library(h2o) # for UMAP, but takes a while so ignore unless necessary

# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
#setwd("C:/Users/oaw001/Documents/AnomalyDetection")

# load in the scripts
scripts <- list("Dictionaries.R", "PlotFunctions.R", "FeatureGeneration.R",
                "FeatureSelection.R", "OtherFunctions.R",
                "UserInput.R", "ModelTuning.R") #, "DataExploration.R")

# Function to source scripts and handle errors
successful <- TRUE
source_script <- function(script) {
  tryCatch(
    source(file.path(base_path, "Scripts", script)), 
    error = function(e) {
      successful <<- FALSE
      message(paste("Error sourcing script:", script))
    }
  )
}
walk(scripts, source_script)

#---------------------------------------------------------------------------
# Define parameters and create data splits for this particular run       ####
#---------------------------------------------------------------------------

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
    unique_ids <- unique(move_data$ID)
    test_ids <- sample(unique_ids, ceiling(length(unique_ids) * test_proportion))
    data_test <- move_data[ID %in% test_ids]
    data_other <- move_data[!ID %in% test_ids]
  # save these
    fwrite(data_test, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_test.csv")))
    fwrite(data_other, file.path(base_path, "Data/Hold_out_test", paste0(movement_data$name, "_other.csv")))
  }

# explore data in PreProcessingDecisions.R to determine window length and behavioural clustering
# haven't automated this yet
# when complete, add to dictionary

#---------------------------------------------------------------------------
# Feature Generation and Elimination                                     ####
#---------------------------------------------------------------------------
# generate all features, unless that has already been done 
#(it can be very time consuming so I tend to save it when I've done it)

if (file.exists(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))){
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
} else {
  feature_data <- generate_features(movement_data, data = data_other, 
                                    normalise = "z_scale", features = features_type)
}

#---------------------------------------------------------------------------
# Tuning model hyperparameters                                           ####
#---------------------------------------------------------------------------
# this section of the code iterates through hyperparameter combinations for OCC
# PR-ROC for the target class is recorded and saved in the output table

# create all the options # options_df exists in a different script 'OtherFunctions.R'
options_df <- expand_all_options(model_hyperparameters_list, feature_hyperparameters_list,
                                             targetActivity_options, model_options, 
                                             feature_selection_method, feature_normalisation_options, 
                                             nu_options, kernel_options, degree_options)

options_row <- options_df[1:3,]

# example data for getting this working
#subset_data <- feature_data %>% group_by(ID, Activity) %>% slice(1:20) %>%ungroup() %>%setDT()

subset_data <- feature_data

# tune the model design by trialing each line in the extended_options_df2
model_outcomes <- map_dfr(1:nrow(options_df), ~process_row(options_df[., ], 
                                                                    k_folds, 
                                                                    subset_data, 
                                                                    validation_proportion, 
                                                                    feature_selection_method, 
                                                                    base_path, 
                                                                    dataset_name, 
                                                                    number_features))

# average these cross-validation results
average_model_outcomes <- model_outcomes %>%
    group_by(across(-PR_AUC)) %>%  
    mutate(PR_AUC = as.numeric(PR_AUC)) %>%
    summarise(
      mean_PR_AUC = mean(PR_AUC, na.rm = TRUE),  # Calculate the mean
      sd_PR_AUC = sd(PR_AUC, na.rm = TRUE)       # Calculate the standard deviation between folds
    )

# save to
average_model_outcomes <- setDT(average_model_outcomes)
ensure.dir(file.path(base_path, "Output", dataset_name))
fwrite(average_model_outcomes, file.path(base_path, "Output", dataset_name, paste0(dataset_name, "_model_outcomes_test3.csv")))





















---------------------------------------------------------------------------
# Testing highest performing model hyperparameters                       ####
---------------------------------------------------------------------------

# load in the validation data and generate features
testing_data <- fread(file.path(base_path, "Data", "Hold_out_test", paste0(dataset_name, "_Labelled_test.csv")))
testing_data_sample <- testing_data %>% group_by(ID, Activity) %>% slice(1:100) %>% ungroup() %>% setDT()

testing_feature_data <- generate_features(movement_data, data = testing_data, 
                                          normalise = "z_scale", features_type = features_type)
# save this for later
fwrite(testing_feature_data, file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))
testing_feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_test_features.csv")))

  
  
# create optimal_row
average_model_outcomes <- setDT(average_model_outcomes)
optimal_rows <- average_model_outcomes[order(-mean_PR_AUC), .SD[1], by = "Activity"]
optimal_row <- optimal_rows[Activity == 'Eating']

# load in training data and select features and target data
training_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_labelled_features.csv")))
selected_feature_data <- feature_selection(training_data, optimal_row)
target_selected_feature_data <- selected_feature_data[Activity == as.character(optimal_row$Activity),
                                                   !label_columns, with = FALSE] 

# create the optimal SVM
params <- list(gamma = optimal_row$gamma, degree = NA) %>% compact()
params <- Filter(Negate(is.na), params)
optimal_single_class_SVM <- do.call(svm, c(list(target_selected_feature_data, y = NULL, type = 'one-classification', 
                                                nu = optimal_row$nu, scale = TRUE, 
                                                kernel = optimal_row$kernel), params))
  
# somehow get the top_features back out so it doesn't need to be extracted like this
top_features <- colnames(target_selected_feature_data)
selected_testing_data <- testing_feature_data[, .SD, .SDcols = c("Activity", top_features)]
selected_testing_data <- selected_testing_data[complete.cases(selected_testing_data), ]

# apply the SVM to the test data 
ground_truth_labels <- selected_testing_data[, "Activity"]
ground_truth_labels <- ifelse(ground_truth_labels$Activity == as.character(optimal_row$Activity), 1, -1)

numeric_testing_data <- selected_testing_data[, !"Activity"]
decision_scores <- predict(optimal_single_class_SVM, newdata = numeric_testing_data, decision.values = TRUE)
scores <- as.numeric(attr(decision_scores, "decision.values"))

# Calculate AUC
roc_curve <- roc(as.vector(ground_truth_labels), scores)
auc_value <- auc(roc_curve)

plot(roc_curve, col = "blue", main = paste("ROC Curve (AUC =", round(auc_value, 2), ")"))

pr_curve <- pr.curve(scores.class0 = scores[ground_truth_labels == 1],
                     scores.class1 = scores[ground_truth_labels == -1], curve = TRUE)
pr_auc_value <- pr_curve$auc.integral
plot(pr_curve)


confusionMatrix(data = factor(predicted_classes, levels = c(-1, 1)), reference = factor(ground_truth_labels, levels = c(-1, 1)))


# find the threshold for classification that maximises F score
thresholds <- seq(0, 1, by = 0.01) # Thresholds to trial

metrics <- function(threshold) {
  # Assign predicted classes based on threshold
  predicted_classes <- ifelse(scores > threshold, 1, -1)
  
  # Create confusion matrix, ensuring that both classes (-1 and 1) are represented
  confusion_matrix <- table(factor(predicted_classes, levels = c(-1, 1)),
                            factor(ground_truth_labels, levels = c(-1, 1)))
  
  # Extract values from confusion matrix or set to 0 if they don't exist
  TP <- ifelse("1" %in% rownames(confusion_matrix) && "1" %in% colnames(confusion_matrix),
               confusion_matrix["1", "1"], 0)
  FP <- ifelse("1" %in% rownames(confusion_matrix) && "-1" %in% colnames(confusion_matrix),
               confusion_matrix["1", "-1"], 0)
  FN <- ifelse("-1" %in% rownames(confusion_matrix) && "1" %in% colnames(confusion_matrix),
               confusion_matrix["-1", "1"], 0)
  
  # Calculate precision and recall with safeguards
  precision <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
  recall <- ifelse(sum(ground_truth_labels == 1) == 0, 0, TP / sum(ground_truth_labels == 1))
  
  # Calculate F1 score
  F1_score <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  
  return(c(threshold, F1_score))
}

# Compute accuracy for each threshold
results <- sapply(thresholds, metrics) %>% as.data.frame()
results <- as.data.frame(t(results))

plot(results$V1, results$V2, xlab = "Threshold", ylab = "F1Score")

# Get the best threshold 
best_threshold <- results$V1[which.max(results$V2)]
