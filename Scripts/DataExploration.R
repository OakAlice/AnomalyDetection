# Plotting the features

# plot the training_data features ####
all_axes <- c("Accelerometer.X", "Accelerometer.Y", "Accelerometer.Z")

feature_distribution_1 <- function(data_processed) {
  # Reshape data from wide to long format
  data_long <- data_processed %>%
    pivot_longer(
      cols = starts_with("X_") | starts_with("Y_") | starts_with("Z_"),
      names_to = "Feature",
      values_to = "Value"
    )
  
  # Create the plot
  ggplot(data_long, aes(x = Value, fill = Activity)) +
    geom_histogram(position = "identity", alpha = 0.6, binwidth = 0.1) +
    facet_wrap(~ Feature, scales = "free_x", ncol = 4) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),  # Remove x-axis text
      axis.text.y = element_blank(),  # Remove y-axis text
      axis.title.x = element_blank(), # Remove x-axis title
      axis.title.y = element_blank(), # Remove y-axis title
      axis.ticks = element_blank()    # Remove axis ticks
    ) +
    labs(
      fill = "Activity"
    )
}


# same idea as above but different view ####
data_long <- data_processed %>%
  pivot_longer(cols = -c(Activity, Timestamp, ID), names_to = "feature", values_to = "value")


data_long_ind <- data_long %>% filter(ID %in% 22)

# Plot using ggplot
ggplot(data_long_ind, aes(x = Activity, y = value, colour = Activity)) +
  geom_jitter() +
  facet_wrap(~ feature, scales = "free_y") +
  theme(axis.text.x = element_blank()) +
  theme_minimal()+
  labs(title = "Jitter Plot of Features by Activity",
       x = "Activity",
       y = "Value")


# interactions between the features ####
library(GGally)

# Select the numeric columns for pairwise plotting
numeric_data <- data_processed %>%
  select(-c(Activity, Timestamp, ID
            )) %>%
  filter_all(all_vars(is.finite(.))) %>%
  na.omit()

# Pairwise scatter plots using ggpairs
numeric_columns <- colnames(numeric_data)

# Create pairwise scatter plots
for (i in 1:(length(numeric_columns) - 1)) {
  for (j in (i + 1):length(numeric_columns)) {
    p <- ggplot(data_processed, aes_string(x = numeric_columns[i], y = numeric_columns[j], colour = "Activity")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(title = paste("Scatter Plot of", numeric_columns[i], "vs", numeric_columns[j]),
           x = numeric_columns[i],
           y = numeric_columns[j])
    
    print(p)
  }
}



## Here ####
library(patchwork)
data_long <- data_processed %>%
  select(-c(Timestamp, ID)) %>%
  gather(key = "feature", value = "value", -Activity)

features <- unique(data_long$feature)

# Initialize an empty plot object
plot_list <- list()

# Iterate through each pair of features
for(i in 1:(length(features)-1)) {
  for(j in (i+1):length(features)) {
    plot <- ggplot(data_processed, aes_string(x = features[i], y = features[j], colour = "Activity")) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      labs(x = features[i], y = features[j]) +
      theme(legend.position = "none")
    
    plot_list[[paste(features[i], features[j], sep = "_vs_")]] <- plot
  }
}

# Combine all plots into a grid using patchwork
combined_plot <- wrap_plots(plot_list, ncol = 3) +
  plot_layout(guides = "collect") +
  plot_annotation(title = "Pairwise Scatter Plots of Features by Activity")

# Display the plot
print(combined_plot)









# how behaviours change over time ####

# plot how the behaviours change over time
data2 <- data_other %>% arrange(ID, Timestamp) %>% slice_head(n = 940000) %>% ungroup() %>%
  mutate(numeric_activity = as.numeric(factor(Activity)), 
         Activity = factor(Activity),
         relative_seconds = row_number())

ggplot(data2, aes(x = (relative_seconds/20), y = as.numeric(numeric_activity))) +
  geom_line() +
  theme_minimal() +
  labs(
    x = "Time (seconds)",
    y = "Activity",
    color = "Activity"
  ) +
  scale_y_continuous(
    breaks = unique(data2$numeric_activity),
    labels = levels(data2$Activity)
  )



# look at the trace shapes ####
beh_trace_plot <- plot_behaviours(behaviours = unique(data_other$Activity), data = data_other, n_samples = 200, n_col = 2)






# Feature discovery ####
data_other <- fread(file.path(base_path, "Data", "Hold_out_test", "Vehkaoja_2018_other.csv")) %>%
  mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                           "Stationary",  Activity))

data_test <- fread(file.path(base_path, "Data", "Hold_out_test", "Vehkaoja_2018_test.csv")) %>%
  mutate(Activity = ifelse(Activity %in% c("Panting", "Sitting", "Lying chest"), 
                           "Stationary",  Activity))

data_test<- create_windows(data_test, 200)


#keras::install_keras()
library(keras)
library(dplyr)
h2o.init()

# Select relevant columns for the autoencoder
data <- create_format_data(training_data = data_other, targetActivity = "Galloping", window_size = 200)
data_autoencoder <- data$target_windows


# Define the input shape
input_dim <- ncol(data_autoencoder)
encoding_dim <- 2  # Number of features to discover

# Input layer
input_layer <- layer_input(shape = c(input_dim))

# Encoder layer
encoder <- input_layer %>%
  layer_dense(units = encoding_dim, activation = 'relu') # can change the function

# Decoder layer
decoder <- encoder %>%
  layer_dense(units = input_dim, activation = 'sigmoid') # and this one

# Define the autoencoder model
autoencoder <- keras_model(inputs = input_layer, outputs = decoder)

# Compile the model
autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)


# Fit the autoencoder to the data
data_matrix <- as.matrix(as.data.frame(data_autoencoder))
history <- autoencoder %>% fit(
  x = data_matrix,      # Input data
  y = data_matrix,      # Target data is the same as input for an autoencoder
  epochs = 50,
  batch_size = 32,
  shuffle = TRUE,
  validation_split = 0.2
)


# apply it to the validation data to see what happens
data_matrix_test <- as.matrix(as.data.frame(data_test))
predicted <- autoencoder %>% predict(data_matrix_test)


# Define the encoder model
encoder_model <- keras_model(inputs = input_layer, outputs = encoder)

# Get the encoded features
encoded_features <- encoder_model %>% predict(data_autoencoder)

# Add the new features to your original data
data_with_features <- data_other %>%
  mutate(Feature1 = encoded_features[, 1],
         Feature2 = encoded_features[, 2])

# View the data with the new features
head(data_with_features)
