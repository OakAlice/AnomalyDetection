# Data Exploration --------------------------------------------------------
if (exists("exploration")) {
  tryCatch(
    {
      # Knit the ExploreData.Rmd file as a PDF and save it to the output folder in base dir
      rmarkdown::render(
        input = file.path(base_path, "Scripts", "ExploreData.Rmd"),
        output_format = "pdf_document",
        output_file = paste0(dataset_name, "_exploration.pdf"), # Only file name here
        output_dir = file.path(base_path, "Output"), # Use output_dir for the path
        params = list(
          base_path = base_path,
          dataset_name = dataset_name,
          sample_rate = sample_rate
        )
      )
      message("Exploration PDF saved to: ", output_file)
    },
    error = function(e) {
      message("Error in making the data exploration pdf: ", e$message)
      stop()
    }
  )
} else {
  print("this script hasn't been set up yet")
}




# Set up the human data ---------------------------------------------------

features_values <- fread(file.path(base_path, "Data", "Anguita_Human", "UCI HAR Dataset", "UCI HAR Dataset", "test", "X_test.txt"))
features_names <- fread(file.path(base_path, "Data", "Anguita_Human", "UCI HAR Dataset", "UCI HAR Dataset", "features.txt"))
names(features_values) <- features_names$V2

ID <- fread(file.path(base_path, "Data", "Anguita_Human", "UCI HAR Dataset", "UCI HAR Dataset", "test", "subject_test.txt"))
names(ID) <- "ID"

labels_numeric <- fread(file.path(base_path, "Data", "Anguita_Human", "UCI HAR Dataset", "UCI HAR Dataset", "test", "y_test.txt"))
labels_words <- fread(file.path(base_path, "Data", "Anguita_Human", "UCI HAR Dataset", "UCI HAR Dataset", "activity_labels.txt"))
labels <- merge(labels_numeric, labels_words, by = "V1")
  
dataset <- cbind(features_values, ID, "Activity" = labels$V2)

fwrite(dataset, file.path(base_path, "Data", "Feature_data", "Anguita_Human_all_test_features.csv"))
