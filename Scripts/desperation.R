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



