# ---------------------------------------------------------------------------
# Assorted functions
# ---------------------------------------------------------------------------

# ensure a directory exists
ensure.dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
}

# Function to apply column selection changes to both training and testing data
update_feature_data <- function(data, multi) {
  cols_to_remove <- c("Activity", "GeneralisedActivity", "OtherActivity")
  # classes to remove logic
  if (multi == "OtherActivity") {
    col_to_rename <- "OtherActivity"
  } else if (multi == "GeneralisedActivity") {
    col_to_rename <- "GeneralisedActivity"
  } else if (multi == "Activity") {
    col_to_rename <- "Activity"
  }

  data <- data %>%
    select(-(setdiff(cols_to_remove, col_to_rename))) %>%
    rename(Activity = col_to_rename)

  return(data)
}


# check which packages are used from SibyllWang on Stack Exchange
checkPacks <- function(path) {
  # Get all R files in the directory
  # Note: Use the pattern ".R$" to match only files ending in .R
  files <- list.files(path)[str_detect(list.files(path), ".R$")]

  # Extract all functions and determine the package they come from using NCmisc::list.functions.in.file
  funs <- unlist(lapply(paste0(path, "/", files), list.functions.in.file))

  # Get the function names (which include package info)
  packs <- funs %>% names()

  # Identify "character" functions such as reactive objects in Shiny
  characters <- packs[str_detect(packs, "^character")]

  # Identify user-defined functions from the global environment
  globals <- packs[str_detect(packs, "^.GlobalEnv")]

  # Identify functions that are in multiple packages' namespaces
  multipackages <- packs[str_detect(packs, ", ")]

  # Extract unique package names from the multipackages
  mpackages <- multipackages %>%
    str_extract_all(., "[a-zA-Z0-9]+") %>%
    unlist() %>%
    unique()

  # Remove non-package elements from mpackages
  mpackages <- mpackages[!mpackages %in% c("c", "package")]

  # Identify functions that are from single packages
  packages <- packs[str_detect(packs, "package:") & !packs %in% multipackages] %>%
    str_replace(., "[0-9]+$", "") %>%
    str_replace(., "package:", "")

  # Get unique packages
  packages_u <- packages %>%
    unique() %>%
    union(., mpackages)

  # Return list of unique packages and their frequency table
  return(list(packs = packages_u, tb = table(packages)))
}
