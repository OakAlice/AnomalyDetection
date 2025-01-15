# Trialling on the iris dataset -------------------------------------------


# Set up ------------------------------------------------------------------
set.seed(42)

# Create train/test split
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

# Train and test multiclass decision tree ---------------------------------
class_weights <- table(train_data$Species)
class_weights <- max(class_weights) / class_weights
weights_vector <- class_weights[train_data$Species]

trained_tree <- rpart(
  formula = Species ~ .,
  data = as.data.frame(train_data),
  method = "class",  # Use "class" for classification
  weights = weights_vector,
  control = rpart.control(
    maxdepth = 30,          
    minsplit = 10, 
    minbucket = 5, 
    cp = 0.01,                
    xval = 10                    
  )
)

truth_labels <- test_data$Species
numeric_pred_data <- as.data.table(test_data)[, !("Species"), with = FALSE]

# decision tree version
prediction_labels <- predict(trained_tree, newdata = as.data.frame(numeric_pred_data), type = "class")

# Create factors with consistent levels
unique_classes <- sort(unique(c(prediction_labels, truth_labels)))
prediction_labels <- factor(prediction_labels, levels = unique_classes)
ground_truth_labels <- factor(truth_labels, levels = unique_classes)

# table(ground_truth_labels, prediction_labels)

# Calculate metrics
macro_multiclass_scores <- multiclass_class_metrics(ground_truth_labels, prediction_labels)
macro_metrics <- macro_multiclass_scores$weighted_metrics

multiclass_F1 <- macro_metrics$F1_Score

# Train and test binary models --------------------------------------------
binary_scores <- list()

flower_species <- unique(iris$Species)
for (species in flower_species){
  
  species_train_data <- train_data %>%
    mutate(Species = ifelse(Species == species, species, "Other"))
  species_test_data <- test_data %>%
    mutate(Species = ifelse(Species == species, species, "Other"))
  
  trained_tree <- rpart(
    formula = Species ~ .,
    data = as.data.frame(species_train_data),
    method = "class",  # Use "class" for classification
    weights = weights_vector,
    control = rpart.control(
      maxdepth = 30,          
      minsplit = 10, 
      minbucket = 5, 
      cp = 0.01,                
      xval = 10                    
    )
  )
  
  truth_labels <- species_test_data$Species
  numeric_pred_data <- as.data.table(species_test_data)[, !("Species"), with = FALSE]
  
  # decision tree version
  prediction_labels <- predict(trained_tree, newdata = as.data.frame(numeric_pred_data), type = "class")
  
  # Create factors with consistent levels
  unique_classes <- sort(unique(c(as.character(prediction_labels), truth_labels)))
  prediction_labels <- factor(prediction_labels, levels = unique_classes)
  ground_truth_labels <- factor(truth_labels, levels = unique_classes)
  
  # table(ground_truth_labels, prediction_labels)
  
  f1_score <- MLmetrics::F1_Score(
    y_true = truth_labels,
    y_pred = prediction_labels,
    positive = species
  )
  
  binary_scores[species] <- f1_score
}

binary_F1 <- mean(unlist(binary_scores))


# Print out the findings --------------------------------------------------
print("multiclass F1 score: ", multiclass_F1)
print("binary F1 score: ", binary_F1)


