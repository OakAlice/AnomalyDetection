# ---------------------------------------------------------------------------
# One Class Classification on Animal Accelerometer Data 3                ####
# ---------------------------------------------------------------------------
# Hyperparmeter Optimisation, one-class and multi-class

#base_path <- "C:/Users/oaw001/Documents/AnomalyDetection"
base_path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection"
source(file.path(base_path, "Scripts", "SetUp.R"))

# Tuning OCC model hyperparameters --------------------------------------
# PR-AUC for the target class is optimised

# Define your bounds for Bayesian Optimization
bounds <- list(
  nu = c(0.001, 0.1),
  gamma = c(0.001, 0.1),
  kernel = c(1, 2, 3),
  number_trees = c(100, 500),
  number_features = c(10, 75)
)

for (activity in target_activities) {
  print(activity)
  
  feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv")))
  feature_data <- feature_data %>% select(-c("OtherActivity", "GeneralisedActivity")) %>% as.data.table()
  
  # Run the Bayesian Optimization
  results <- BayesianOptimization(
    FUN = function(nu, gamma, kernel, number_trees, number_features) {
      modelTuning(
        feature_data = feature_data,
        target_activity = activity, 
        nu = nu,
        kernel = kernel,
        gamma = gamma,
        number_trees = number_trees,
        number_features = number_features
      )
    },
    bounds = bounds,
    init_points = 5,
    n_iter = 10,
    acq = "ucb",
    kappa = 2.576 
  )
}
#}

# Tuning multiclass model hyperparameters ---------------------------------
# this section of the code tunes the multiclass model, optimising macro average PR-AUC
# the same bounds as in the above section are used for comparison sake
# remmeber to account for there being multiple types of activity columns 
behaviour_columns <- c("Activity", "OtherActivity", "GeneralisedActivity")

feature_data <- fread(file.path(base_path, "Data", "Feature_data", paste0(dataset_name, "_multi_features.csv")))
feature_data <- feature_data %>% as.data.table()

if (tuningMulti == TRUE){
  #for (behaviours in behaviour_columns){
    
    behaviours <- "OtherActivity"
    
    multiclass_data <- feature_data %>%
      select(-(setdiff(behaviour_columns, behaviours))) %>%
      rename("Activity" = !!sym(behaviours)) #%>%
      #select(-"...6") # remove this random column
    
    if(behaviours == "GeneralisedActivity"){
      multiclass_data <- multiclass_data %>% filter(!Activity == "")
    }
    
    # Run the Bayesian Optimization
    results <- BayesianOptimization(
      FUN = function(nu, gamma, kernel, number_trees, number_features) {
        multiclassModelTuning(
          multiclass_data = multiclass_data,
          nu = nu,
          kernel = kernel,
          gamma = gamma,
          number_trees = number_trees,
          number_features = number_features
        )
      },
      bounds = bounds,
      init_points = 5,
      n_iter = 10,
      acq = "ucb",
      kappa = 2.576 
    )
#  }
}
