# Data Exploration --------------------------------------------------------
# need to check whether this works yet
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
