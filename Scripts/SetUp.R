# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data                  ####
# ---------------------------------------------------------------------------

# PART ONE: SET UP ----------------------------------------------------------
# script mode
# mark TRUE what stage you want to execute
# exploration for generating PDF, tuning for HPO finding, testing for final validation
exploration   <- FALSE
renamed       <- TRUE # whether I have already added in the general beh categories

# User Defined Variables ---------------------------------------------------
# set base path/directory from where scripts, data, and output are stored
base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
dataset_name <- "Ladds_Seal"
#dataset_name <- "Vehkaoja_Dog"
sample_rate <- 25

# install.packages("pacman")
library(pacman)
p_load(
  bench, caret, data.table, e1071, future, future.apply, parallelly,
  plotly, PRROC, purrr, pROC, rBayesianOptimization,
  randomForest, tsfeatures, tidyverse, umap, zoo, tinytex
)
# note that tinytex needs this too -> tinytex::install_tinytex()
#library(h2o) is for UMAP, but takes a while so ignore unless necessary

# some other things I need defined globally :'(
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")
label_columns <- c("Activity", "Time", "ID")
test_proportion <- 0.2
validation_proportion <- 0.2
features_type <- c("timeseries", "statistical")

# load in the function scripts
function_scripts <-
  list(
    "Scripts/Functions/BaselineSVMFunctions.R",
    "Scripts/Functions/FeatureGenerationFunctions.R",
    "Scripts/Functions/FeatureSelectionFunctions.R",
    "Scripts/Functions/OtherFunctions.R",
    "Scripts/Functions/ModelTuningFunctions.R",
    "Scripts/Functions/CalculatePerformanceFunctions.R"
  )

# Function to source scripts and handle errors
successful <- TRUE
source_script <- function(script) {
  tryCatch(
    source(file.path(base_path, script)),
    error = function(e) {
      successful <<- FALSE
      message(paste("Error sourcing script:", script))
    }
  )
}
walk(function_scripts, source_script)

# Split Test Data ---------------------------------------------------------
move_data <- fread(file.path(base_path, "Data", paste0(dataset_name, ".csv")))

# Split Data ####
if (file.exists(file.path(
  base_path, "Data", "Hold_out_test", paste0(dataset_name, "_test.csv")
))) {
  # if this has been run before, just load in the split data
  data_test <-
    fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  data_other <-
    fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")))
} else {
  # if this is the first time running code for this dataset, create hold-out test set
  unique_ids <- unique(move_data$ID)
  test_ids <-
    sample(unique_ids, ceiling(length(unique_ids) * test_proportion))
  data_test <- move_data[ID %in% test_ids]
  data_other <- move_data[!ID %in% test_ids]
  # save these
  fwrite(data_test,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_test.csv")))
  fwrite(data_other,
         file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_other.csv")
         ))
}

# Data Exploration --------------------------------------------------------
# need to check whether this works yet
if (exploration == TRUE) {
  tryCatch({
    # Knit the ExploreData.Rmd file as a PDF and save it to the output folder in base dir
    rmarkdown::render(
      input = file.path(base_path, "Scripts", "ExploreData.Rmd"),
      output_format = "pdf_document",
      output_file = paste0(dataset_name, "_exploration.pdf"),  # Only file name here
      output_dir = file.path(base_path, "Output"),  # Use output_dir for the path
      params = list(
        base_path = base_path,
        dataset_name = dataset_name,
        sample_rate = sample_rate
      )
    )
    message("Exploration PDF saved to: ", output_file)
  }, error = function(e) {
    message("Error in making the data exploration pdf: ", e$message)
    stop()
  })
}











