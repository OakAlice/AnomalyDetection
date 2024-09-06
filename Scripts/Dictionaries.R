# Dictionaries

# list formatted data
# formatted means columns: Time, ID, Accelerometer.X, Acceleormeter.Y, Accelerometer.Z, Activity
all_dictionaries <- list("Vehkaoja_Dog" = "Vehkaoja_Dog_Labelled",
                         "Studd_Squirrel"="Studd_Squirrel_Labelled",
                         "Annett_Possum" = "Annett_Possum_Labelled",
                         "Ladds_Seal" = "Ladds_Seal_Labelled",
                         "Pagano_Bear" = "Pagano_Bear_Labelled",
                         "Yu_Duck" = "Yu_Duck_Labelled",
                         "Smit_Cat" = "Smit_Cat_Labelled"
                         )



Vehkaoja_Dog_Labelled <- list(name = "Vehkaoja_Dog_Labelled",
                              "Frequency" = 100, 
                              target_behaviours = c("Eating", "Sitting"), # regroup these to stationary
                              window_length = 1,
                              overlap_percent = 0
                              )

Studd_Squirrel_Labelled <- list(name = "Studd_Squirrel_Labelled",
                                "Frequency" = 1, 
                                target_behaviours = c("Feeding", "Stationary")
                                )

Annett_Possum_Labelled <- list("Frequency" = 50, target_behaviours = c())
Ladds_Seal_Labelled <- list("Frequency" = 25, target_behaviours = c())
Pagano_Bear_Labelled <- list("Frequency" = 16, target_behaviours = c())
Yu_Duck_Labelled <- list("Frequency" = 25, target_behaviours = c())
Smit_Cat_Labelled <- list("Frequency" = 30, target_behaviours = c())









#exploration_dataset <- exploration_dataset %>% 
#  select("DogID", "t_sec", "ANeck_x", "ANeck_y", "ANeck_z", "Behavior_2") %>%
#  rename("ID" = "DogID",
#         "Time" = "t_sec", 
#         "Accelerometer.X" = "ANeck_x", 
#         "Accelerometer.Y" ="ANeck_y", 
#         "Accelerometer.Z" = "ANeck_z", 
#         "Activity" = "Behavior_2")
#exploration_dataset <- exploration_dataset %>% filter(!Activity == "<undefined>")

#fwrite(exploration_dataset, file.path(base_path, "Data", paste0(exploration_dataset_name, ".csv")))
