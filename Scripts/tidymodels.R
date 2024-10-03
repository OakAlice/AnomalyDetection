# Splitting data into training and validation sets
data_split <- initial_split(subset_data, prop = 1 - validation_proportion, strata = ID)
training_data <- training(data_split)
validation_data <- testing(data_split)

# Recipe for feature selection and normalization
data_recipe <- recipe(Activity ~ ., data = training_data) %>%
  step_mutate(Activity = ifelse(Activity == options_row$targetActivity, "Target", "Other")) %>%
  step_select(all_predictors()) %>%
  step_normalize(all_predictors())  # Normalize features


svm_model <- svm_rbf(cost = options_row$nu, rbf_sigma = options_row$gamma) %>%
  set_engine("kernlab") %>%
  set_mode("classification")



svm_workflow <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(data_recipe)



# Set up cross-validation folds
folds <- vfold_cv(training_data, v = k_folds, strata = ID)

# Perform grid tuning
svm_grid <- grid_regular(cost(range = c(0.05, 0.2)), rbf_sigma(range = c(0.01, 0.1)), levels = 5)

# Train the model with tuning
svm_tune <- tune_grid(
  svm_workflow,
  resamples = folds,
  grid = svm_grid,
  metrics = metric_set(roc_auc, pr_auc, accuracy, precision, recall, f_meas)
)


# Collect best results
best_results <- collect_metrics(svm_tune)

# Extract AUC and other relevant metrics
auc_value <- best_results %>% filter(.metric == "roc_auc") %>% pull(mean)
pr_auc_value <- best_results %>% filter(.metric == "pr_auc") %>% pull(mean)



plan(multisession, workers = parallel::detectCores())

# Fit the model with parallel cross-validation
svm_results <- tune_grid(
  svm_workflow,
  resamples = folds,
  grid = svm_grid
)

plan(sequential)  # Switch back to sequential after tuning



final_workflow <- finalize_workflow(svm_workflow, select_best(svm_tune, "roc_auc"))

final_model <- fit(final_workflow, data = training_data)


predictions <- predict(final_model, new_data = validation_data)

# Calculate accuracy and other metrics on validation set
validation_metrics <- validation_data %>%
  bind_cols(predictions) %>%
  metrics(truth = Activity, estimate = .pred_class)
