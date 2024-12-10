# Adding activity category columns ----------------------------------------

for (condition in c("other", "test")) {
  data <- fread(file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
  new_column_data <- renameColumns(data, dataset_name, target_activities)
  fwrite(new_column_data, file.path(base_path, "Data/Hold_out_test", paste0(dataset_name, "_", condition, ".csv")))
}

# add the columns to categroies
renameColumns <- function(data, dataset_name, target_activities) {
  if (dataset_name == "Vehkaoja_Dog") {
    # 4 general categories, discarding difficult behaviours
    data[, GeneralisedActivity := fifelse( # nested ifelse
      Activity %in% c("Walking", "Trotting", "Pacing", "Tugging", "Jumping", "Galloping", "Carrying object"), "Travelling",
      fifelse(
        Activity %in% c("Eating", "Drinking", "Sniffing"), "Feeding",
        fifelse(
          Activity %in% c("Sitting", "Lying chest", "Standing"), "Resting",
          fifelse(Activity == "Shaking", "Grooming", NA_character_)
        )
      )
    )]
    # 4 specific categories and a non-specific "other"
    data[, OtherActivity := ifelse(Activity %in% target_activities, Activity, "Other")]
  } else if (dataset_name == "Ladds_Seal") {
    # 4 general categories, discarding difficult behaviours
    data[, GeneralisedActivity := fifelse( # nested ifelse
      Activity %in% c("swimming", "sailing", "slow", "fast", "moving"), "Swimming",
      fifelse(
        Activity %in% c("chewing", "holdntear", "feeding", "manipulation"), "Chewing",
        fifelse(
          Activity %in% c("still", "lying", "sitting", "stationary"), "Still",
          fifelse(Activity %in% c("scratch", "rubbing", "facerub", "shake", "grooming"), "Facerub", NA_character_)
        )
      )
    )]
    # 4 specific categories and a non-specific "other"
    data[, OtherActivity := ifelse(Activity %in% target_activities, Activity, "Other")]
  } else {
    print("Behavioural categories have not been defined for this dataset")
  }

  return(data)
}
