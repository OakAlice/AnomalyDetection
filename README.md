# Decomposing Complex Problems
Complex classification problems can be decomposed into a series of easier classification problems allowing us to target specific classes and make fewer assumptions about uncertain classes. In theory, an ensemble of these simpler models should perform as well as, or better than, the original multi-class models for open set recongition problems (where there are classes in real set not in training set).
In this code, I explore this possibility in the context of animal accelerometry, working with example datasets of dogs and seals - dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348)and seals sourced from [Ladds et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166898).
I compare 1-class, binary (1-vs-all), and three kinds of multi-class models on their ability to detect 4 specific behaviours from datasets that contain progressively fewer training classes, testing on datasets with all training classes. I trial both an SVM and Random Forest approach.

## Progress Note
In progress. Currently interpretable to me but needs a lot of work to be transferable to others.

## Scripts
### R
There are initialisation scripts that source functions. There is one data exploration RMarkdown file. The remainder of the files are functions.
* [MainScript,R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/MainScript.R) <- Load packages, load functions, set experimental variables, split out test data
* [ExploreData.Rmd](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/ExploreData.Rmd) <- Explore data to determine appropriate target behaviours, window length, etc. When executed, this saved a PDF RMarkdown report with exploratory graphs and tables.
* [Preprocessing.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/Preprocessing.R) <- Define behaviour types and window lengths. Currently set to generate timeseries and statistical features. Add other pre-processing such as filtering and normalisation later.
* [SVMHpoOptimisation.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/SVMHpoOptimisation.R) <- Using Bayesian Optimisation search the hyperparameter space and tune models on k-fold cross-validation between training and validation (Also a file like this for Tree models, and for the next scripts as well)
* [SVMTrainBestModels.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/SVMTrainBestModels.R) <- Train final models with optimal hyperparameter set using all non-test data
* [SVMTestBestModels.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/SVMTestBestModels.R) <- Test all OCC and multi-class models on their test sets
* [EnsembleDevelopment.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/EnsembleDevelopment.R) <- Combine predictions from the 4 individual models into a collective predictions set (not an ensemble in the traditional sense)
* [PlotttingPerformance.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/PlottingPerformance.R) <- Generate PDF plots for the comparative performance of each of the models

### Python
Eventually I switched my workflow over to python to utilise the more advanced ML libraries available there.
* [requirements.txt](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/requirements.txt) <- packages
* [MainScript.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/MainScript.py) <- Define variables
* [CreateDatasets.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/CreateDatasets.py) <- split out data into subframes based on TRAINING_SET, clean data, save to disk for easier retrieval later
* [SVMHpoOptimisation.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/SVMHpoOptimisation.py) <- Bayesian search hyperparameters to identify optimal model variables
* [SVMTrainModel.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/SVMTrainModel.py) <- Train optimal model based on identified optimal hyperparameters
* [SVMTestModel.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/SVMTestModel.py) <- Test model on complete test set

