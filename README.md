# One-Class Classification
Code for a case study demonstrating the use of one-class classification (OCC) for detecting specific target behaviours in animal accelerometer behavioural classification data. Manually input raw labelled data and hyperparameter options, manually select target behaviours and window lengths, and then workflow automatically generates features and tunes model design, tests highest performing design on hold-out set, and calculates final performance for each model and dataset.

Example dataset of 45 dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348). Dataset of 12 seals sourced from [Ladds et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166898).

## Disclaimer
The code is currently a bit of a Frankenstein that I've built bit by bit as a I realised I needed more things. It's probably twice as long as it should be. Additionally, it is hyper-specific to the bugs and characteristsics of the data and use-case I developed it for (i.e., demonstrating viability of OCC), so generalising it to other use cases (e.g., doing OCC) may require pulling the code apart. I will do this later, but not yet.

## Scripts
There is one initialisation script that sources multiple other tasks. There is one data exploration RMarkdown files. The remainder of the files are functions.
* [LocalInitialisationScript](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/LocalInitialisationScript.R) <- Main script from which other scripts are executed, requires some user input to define variables
* [ExploreData](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/ExploreData.Rmd) <- Explore data to determine appropriate target behaviours, window length, etc. When executed, this saved a PDF RMarkdown report with exploratory graphs and tables.
* [FeatureGeneration](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Functions/FeatureGenerationFunctions.R) <- For each window, generate statistical and/or timeseries features
* [FeatureSelection](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Functions/FeatureSelectionFunctions.R) <- Select most discriminatory features with RF or reduce dimensionality with UMAP
* [ModelTuning](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Functions/ModelTuningFunctions.R) <- Using Bayesian Optimisation search the hyperparameter space and tune models on k-fold cross-validation between training and validation
* [Baseline SVM](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Functions/BaselineSVMFunctions.R) <- Create the multiclass models and compare performance

[DatasetModification](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/DogDatasetModification.R) was used for standardising the format of the example data prior to analysis.