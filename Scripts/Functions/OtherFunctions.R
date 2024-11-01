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
  
  data <- data %>% select(-(setdiff(cols_to_remove, col_to_rename))) %>%
  rename(Activity = col_to_rename)
  
  return(data)
}

# add the columns to categroies
renameColumns <- function(data, dataset_name, target_activities){
  
  if (dataset_name == "Vehkaoja_Dog"){
    # 4 general categories, discarding difficult behaviours
    data[, GeneralisedActivity := fifelse( # nested ifelse
      Activity %in% c("Walking", "Trotting", "Pacing", "Tugging", "Jumping", "Galloping", "Carrying object"), "Travelling",
      fifelse(Activity %in% c("Eating", "Drinking", "Sniffing"), "Feeding",
              fifelse(Activity %in% c("Sitting", "Lying chest", "Standing"), "Resting",
                      fifelse(Activity == "Shaking", "Grooming", NA_character_)
              )))]
    # 4 specific categories and a non-specific "other"
    data[, OtherActivity := ifelse(Activity %in% target_activities, Activity, "Other")]
    
  } else if (dataset_name == "Ladds_Seal") {
    # 4 general categories, discarding difficult behaviours
    data[, GeneralisedActivity := fifelse( # nested ifelse
      Activity %in% c("swimming", "sailing", "slow", "fast", "moving"), "Swimming",
      fifelse(Activity %in% c("chewing", "holdntear", "feeding", "manipulation"), "Chewing",
              fifelse(Activity %in% c("still", "lying", "sitting", "stationary"), "Still",
                      fifelse(Activity %in% c("scratch", "rubbing", "facerub", "shake", "grooming"), "Scratch", NA_character_)
              )))]
    # 4 specific categories and a non-specific "other"
    data[, OtherActivity := ifelse(Activity %in% target_activities, Activity, "Other")]
    
  } else {
    print("Behavioural categories have not been defined for this dataset")
  }
  
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
