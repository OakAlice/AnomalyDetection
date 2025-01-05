# Adding activity category columns ----------------------------------------

for (condition in c("other", "test")) {
  data <- fread(file.path(base_path, "Data/Feature_data", paste0(dataset_name, "_", condition, "_features.csv")))
  new_column_data <- renameColumns(data, dataset_name, target_activities)
  fwrite(new_column_data, file.path(base_path, "Data/Feature_data", paste0(dataset_name, "_", condition, "_features.csv")))
}

# add the columns to categroies
renameColumns <- function(data, dataset_name, target_activities) {
  if (dataset_name == "Vehkaoja_Dog") {
    # 4 general categories, discarding difficult behaviours
    data[, GeneralisedActivity := fifelse( # nested ifelse
      Activity %in% c("Walking", "Trotting", "Pacing", "Tugging", "Jumping", "Galloping", "Carrying object"), "Walking",
      fifelse(
        Activity %in% c("Eating", "Drinking", "Sniffing"), "Eating",
        fifelse(
          Activity %in% c("Sitting", "Lying chest", "Standing"), "Lying chest",
          fifelse(Activity == "Shaking", "Shaking", NA_character_)
        )
      )
    )]
    # 4 specific categories and a non-specific "other"
    data[, OtherActivity := ifelse(Activity %in% target_activities, Activity, "Other")]
  } else if (dataset_name == "Ladds_Seal") {
    # 4 general categories, discarding difficult behaviours
    data[, GeneralisedActivity := fifelse( # nested ifelse
      Activity %in% c("swimming", "sailing", "slow", "fast", "moving"), "swimming",
      fifelse(
        Activity %in% c("chewing", "holdntear", "feeding", "manipulation"), "chewing",
        fifelse(
          Activity %in% c("still", "lying", "sitting", "stationary"), "still",
          fifelse(Activity %in% c("scratch", "rubbing", "facerub", "shake", "grooming"), "facerub", NA_character_)
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
