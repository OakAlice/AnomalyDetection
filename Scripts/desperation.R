
# Libraries and set up ----------------------------------------------------
library(data.table)
library(tidyverse)
library(lme4)
library(emmeans)
library(lmerTest)


# Load in the data --------------------------------------------------------
path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection/Output/Combined/all_combined_metrics.csv"
# path <- "C:/Users/oaw001/OneDrive - University of the Sunshine Coast/AnomalyDetection/Output/Combined/all_combined_metrics.csv"
df <- read.csv(path)


# Hypothesis 1 ------------------------------------------------------------
# Performance (AUC) of control models with increasingly open sets
# tested conditions of control model with an linear mixed effects model (with fold as a random effect)

## 1: Closed tests #####
# subset the dataset
subset_df <- df %>% filter(closed_open == "closed",
                             dataset == "Ferdinandy_Dog",
                           behaviour == 'weighted_avg',
                           model_type == 'multi_Activity_NOthreshold')

# factorise and level variables to define the controls
subset_df$training_set <- relevel(factor(subset_df$training_set), ref = "all")
# run the mixed effects model
mixed_model_AUC <- lmer(AUC ~ training_set + (1|fold), data = subset_df)
# Display the summary of the model
summary(mixed_model_AUC)

## 2. Open tests ####
subset_df <- df %>% filter(closed_open == "open",
                           dataset == "Ferdinandy_Dog",
                           behaviour == 'weighted_avg',
                           model_type == 'multi_Activity_NOthreshold')

subset_df$training_set <- relevel(factor(subset_df$training_set), ref = "all")
model <- lmer(AUC ~ training_set + (1|fold), data = subset_df)
summary(model)


# Hypothesis 2 ------------------------------------------------------------
# tested each model type with an linear mixed effects model (with fold as a random effect)

subset_df <- df %>% filter(closed_open == "open",
                           dataset == "Ferdinandy_Dog",
                           behaviour == "weighted_avg")

## Part 1: Overall patterns ####
full_model <- lmer(AUC ~ training_set * model_type + (1 | fold), data = subset_df, REML = FALSE)
reduced_model <- lmer(AUC ~ training_set + model_type + (1 | fold), data = subset_df, REML = FALSE)
anova(reduced_model, full_model)

## Part 2: AUC within each model type ####
subset_df$training_set <- relevel(factor(subset_df$training_set), ref = "all")
model_types <- unique(subset_df$model_type)

for(model in model_types){
  cat("/n=== LME model for Model:", model, "===/n")
  
  subset_df2 <- df %>% filter(model_type == model)
  
  # run the mixed effects model
  mixed_model_AUC <- lmer(AUC ~ training_set + (1|fold), data = subset_df2)
  # anova_AUC <- aov(AUC ~ training_set, data = subset_df)
  
  # Display the summary of the model
  print(summary(mixed_model_AUC))
}

## Part 3. Specificity of all models with increasingly open sets ####
subset_df$training_set <- relevel(factor(subset_df$training_set), ref = "all")
model_types <- unique(subset_df$model_type)

for(model in model_types){
  
  model = 'oneclass'
  print(paste0("=============", model, "=============="))
  
  subset_df2 <- df %>% filter(model_type == model)
  
  # run the mixed effects model
  mixed_model_Spec <- lmer(Specificity ~ training_set + (1|fold), data = subset_df2)
  # anova_AUC <- aov(AUC ~ training_set, data = subset_df)
  
  # Display the summary of the model
  print(summary(mixed_model_Spec))
}

# Hypothesis 3 ------------------------------------------------------------
# Binary models perform best on incomplete models
# which model performs best
subset_df <- df %>% filter(closed_open == "open",
                           dataset == "Ferdinandy_Dog",
                           behaviour == "weighted_avg")

subset_df$training_set <- relevel(factor(subset_df$training_set), ref = "all")
subset_df$model_type <- relevel(factor(subset_df$model_type), ref = "multi_Activity_NOthreshold")

## Approach 1: split trainign sets ####
for (condition in unique(subset_df$training_set)){
  cat("/n======= LME set for condtiion:", condition, "=======/n")
  
  subset_df2 <- subset_df %>% filter(training_set == condition)
  
  linear_fit <- lmer(AUC ~ model_type + (1 | fold), data = subset_df2)
  print(summary(linear_fit))
}

## Approach 2: combine training sets ####
full_model <- lmer(AUC ~ training_set * model_type + (1 | fold), data = subset_df, REML = FALSE)

# Get ANOVA table to test significance of interaction
anova(full_model)

# Post-hoc pairwise comparisons for model type within each training set
emmeans(full_model, pairwise ~ model_type | training_set, adjust = "bonferroni")







path <- "C:/Users/PC/OneDrive - University of the Sunshine Coast/AnomalyDetection/Data/Feature_data/Ferdinandy_Dog_features.csv"
data <- fread(path)

counts <- data %>% group_by(Activity) %>% count()  


  
  
  