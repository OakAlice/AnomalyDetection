# training_set <- "all"

# Read all files together -------------------------------------------------
test_files <- list.files(file.path(base_path, "Output", "Testing", ML_method), pattern = paste0(dataset_name, "_", training_set, "_.*\\.csv$"), full.names = TRUE)
test_outcome <- rbindlist(
  lapply(test_files, function(file) {
    df <- fread(file)
    return(df)
  }),
  use.names = TRUE, fill=TRUE
)

test_outcome[is.na(test_outcome)] <- 0

test_outcome$Activity <- str_to_title(test_outcome$Activity) # format for consistency
# calculate the adjusted values
combined_results_adjusted <- test_outcome %>%
  mutate(Zero_adj_F1_Score = F1_Score - ZeroR_F1_Score,
         Zero_adj_Precision = Precision - ZeroR_Precision,
         Zero_adj_Recall = Recall - ZeroR_Recall,
         Zero_adj_Accuracy = Accuracy - ZeroR_Accuracy,
         Rand_adj_F1_Score_prev = F1_Score - Random_F1_Score_prev,
         Rand_adj_Precision_prev = Precision - Random_Precision_prev,
         Rand_adj_Recall_prev = Recall - Random_Recall_prev,
         Rand_adj_Accuracy_prev = Accuracy - Random_Accuracy_prev,
         Rand_adj_F1_Score_equal = F1_Score - Random_F1_Score_equal,
         Rand_adj_Precision_equal = Precision - Random_Precision_equal,
         Rand_adj_Recall_equal = Recall - Random_Recall_equal,
         Rand_adj_Accuracy_equal = Accuracy - Random_Accuracy_equal)

fwrite(combined_results_adjusted, file.path(base_path, "Output", "Testing", paste0(dataset_name, "_", training_set, "_", ML_method, "_complete_test_performance.csv")))


# Confusion matrices ------------------------------------------------------
# training_set <- "some"
confusion_files <- list.files(file.path(base_path, "Output", "Testing", ML_method, "Confusion"), 
                              pattern = paste0(dataset_name, "_.*\\.csv$"), 
                              full.names = TRUE)

confusion_outcome <- rbindlist(
  lapply(confusion_files, function(file) {
    # Extract training_set and model from file path
    file_name <- basename(file)  # Get just the filename without path
    parts <- strsplit(file_name, "_")[[1]]
    
    # Read the CSV
    df <- fread(file)
    
    df$training_set <- parts[3]
    df$model <- parts[4]
    
    return(df)
  }),
  use.names = TRUE, fill=TRUE
)

confusion_outcome <- confusion_outcome %>%
  mutate(
    ground_truth_labels = ifelse(is.na(ground_truth_labels), 
                                 collective_ground_truth, 
                                 ground_truth_labels),
    prediction_labels = ifelse(is.na(prediction_labels), 
                               collective_predictions, 
                               prediction_labels)
  ) %>%
  select(!c(collective_ground_truth, collective_predictions))

# save this
fwrite(confusion_outcome, 
       file.path(base_path, "Output", "Testing", 
                 paste0(dataset_name, "_", 
                        ML_method, "_complete_confusion.csv")))

