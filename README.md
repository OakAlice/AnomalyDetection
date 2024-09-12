# AnomalyDetection
Code for a case study demonstrating the use of one-class classification (OCC) for detecting specific target behaviours in animal accelerometer behavioural classification data. Input raw labelled data and hyperparameter options, workflow generates features and tunes model design, tests highest performing design on hold-out set, then calculates final performance for that dataset. Example dataset of 45 dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348).

## Scripts
Dictionaries <- Define dataset specific parameters, such as sampling rate and target behaviours
UserInput <- Define hyperparameters to test
InitialisationScript <- Main script from which other scripts are executed
PreProcessingDecisions <- Explore data to determine appropriate window length for target behaviours
FeatureGeneration <- For each window, generate statistical and/or timeseries features
FeatureSelection <- Select most discriminatory features with RF or reduce dimensionality with UMAP

Other scripts include the PlotFunctions for data exploration, OtherFunctions for assorted stuff, and DogDatasetModification for standardising the format of the example data prior to beginning.