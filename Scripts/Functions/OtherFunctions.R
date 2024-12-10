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