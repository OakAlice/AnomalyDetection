#load in the data
library(data.table)
library(tidyverse)

path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection/Output/Testing/all_combined_metrics.csv"

data <- fread(path)

d1 <- data %>% filter(dataset == "Vehkaoja",
                      training_set == "all")

d2 <- data %>% filter(dataset == "Vehkaoja",
                      training_set == "all",
                      behaviour %in% c("Walking", "Shaking", "Lying chest", "Eating", "Other", "weighted_avg"))

p <- ggplot(d2, aes(x = model_type, y = AUC, colour = behaviour)) +
  geom_point(size = 5, alpha = 0.8)
p





# Load in the data --------------------------------------------------------
path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection/Output/Combined/all_combined_metrics.csv"
df <- read.csv(path)

# Account for random baseline ---------------------------------------------






# Run ANOVA for each training set
training_sets <- unique(df$training_set)

for (train_set in training_sets) {
  cat("\n=== ANOVA for Training Set:", train_set, "===\n")
  
  subset_df <- df %>% filter(closed_open == "open",
                             training_set == train_set)
  
  anova_model <- aov(AUC ~ model_type + dataset + behaviour, data = subset_df)
  print(summary(anova_model))  # ANOVA results
  
  # If ANOVA is significant, run post-hoc Tukey test
  if (summary(anova_model)[[1]][["Pr(>F)"]][1] < 0.05) {
    cat("\nTukey HSD Post-hoc Test:\n")
    tukey_results <- TukeyHSD(anova_model)
    significant_results <- tukey_results$model_type[tukey_results$model_type[, "p adj"] < 0.05, ]
    print(significant_results)
  } else {
    cat("No significant differences between models for this training set.\n")
  }
}

# Run ANOVA for each model
models <- unique(df$model_type)

for (model in models) {
  cat("\n=== ANOVA for Training Set:", model, "===\n")
  
  subset_df <- df %>% filter(closed_open == "open",
                             model_type == model
  )
  
  anova_model <- aov(AUC ~ training_set + dataset + behaviour, data = subset_df)
  print(summary(anova_model))  # ANOVA results
  
  # If ANOVA is significant, run post-hoc Tukey test
  if (summary(anova_model)[[1]][["Pr(>F)"]][1] < 0.05) {
    cat("\nTukey HSD Post-hoc Test:\n")
    tukey_results <- TukeyHSD(anova_model)
    significant_results <- tukey_results$training_set # [tukey_results$training_set[, "p adj"] < 0.05, ]
    print(significant_results)
    
  } else {
    cat("No significant differences between training_sets for this model type.\n")
  }
}



# everything
anova_model <- aov(AUC ~ training_set + model_type + dataset + behaviour, data = df)
print(summary(anova_model))  # ANOVA results

# If ANOVA is significant, run post-hoc Tukey test
if (summary(anova_model)[[1]][["Pr(>F)"]][1] < 0.05) {
  cat("\nTukey HSD Post-hoc Test:\n")
  tukey_results <- TukeyHSD(anova_model)
  print(tukey_results)
} else {
  cat("No significant differences between training_sets for this model type.\n")
}
