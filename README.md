# AnomalyDetection
Code for a case study demonstrating the use of one-class classification (OCC) for detecting specific target behaviours in animal accelerometer behavioural classification data. Input raw labelled data and hyperparameter options, workflow generates features and tunes model design, tests highest performing design on hold-out set, then calculates final performance for that dataset. Example dataset of 45 dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348). Note: Repo is named 'Anomaly Detection' because this was what I was originally calling the technique... Now calling it OCC.

## Scripts
* [Dictionaries](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Dictionaries.R) <- Define dataset specific parameters, such as sampling rate and target behaviours
* [UserInput](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/UserInput.R) <- Define hyperparameters to test
* [InitialisationScript](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/InitialisationScript.R) <- Main script from which other scripts are executed
* [PreProcessingDecisions](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/PreProcessingDecisions.R) <- Explore data to determine appropriate window length for target behaviours
* [FeatureGeneration](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/FeatureGeneration.R) <- For each window, generate statistical and/or timeseries features
* [FeatureSelection](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/FeatureSelection.R) <- Select most discriminatory features with RF or reduce dimensionality with UMAP

Other scripts include the [PlotFunctions](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/PlotFunctions.R) for data exploration, [OtherFunctions](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/OtherFunctions.R) for assorted stuff, and [DogDatasetModification](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/DogDatasetModification.R) for standardising the format of the example data prior to beginning.