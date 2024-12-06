
# Split Test Data ---------------------------------------------------------
if (file.exists(file.path(
  base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")
))) {
  # if this has been run before, just load in the split data
  print("Individuals have been split for this dataset previously.")
} else {
  # if this is the first time running code for this dataset, create hold-out test set
  move_data <- fread(file.path(base_path, "Data", paste0(dataset_name, ".csv")))
  unique_ids <- unique(move_data$ID)
  test_ids <- sample(unique_ids, ceiling(length(unique_ids) * test_proportion))
  data_test <- move_data[ID %in% test_ids]
  data_other <- move_data[!ID %in% test_ids]
  # save these
  fwrite(data_test,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  fwrite(data_other,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")
         ))
}