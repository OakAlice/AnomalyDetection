# One-Class Classification
Code for a case study demonstrating the use of one-class classification (OCC) for detecting specific target behaviours in animal accelerometer behavioural classification data. Input raw labelled data and hyperparameter options, workflow generates features and tunes model design, tests highest performing design on hold-out set, then calculates final performance for that dataset. Example dataset of 45 dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348).

## Scripts
* [UserInput](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/UserInput.R) <- Define parameters and various helper things
* [InitialisationScript](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/InitialisationScript.R) <- Main script from which other scripts are executed
* [PreProcessingDecisions](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/PreProcessingDecisions.R) <- Explore data to determine appropriate target behaviours, window length, etc.
* [FeatureGeneration](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/FeatureGeneration.R) <- For each window, generate statistical and/or timeseries features
* [FeatureSelection](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/FeatureSelection.R) <- Select most discriminatory features with RF or reduce dimensionality with UMAP
* [ModelTuning](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/ModelTuning.R) <- Using Bayesian Optimisation search the hyperparameter space and tune models on k-fold cross-validation between training and validation
* Test final model performance on all labelled data used in model development, hold-out test data, randomised data, as well as on a baseline multi-class SVM

[DatasetModification](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/DogDatasetModification.R) was used for standardising the format of the example data prior to analysis.