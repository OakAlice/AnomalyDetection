# Decomposing Complex Problems
Complex classification problems can be decomposed into a series of easier classification problems allowing us to target specific classes and make fewer assumptions about uncertain classes. In theory, an ensemble of these simpler models should perform as well as, or better than, the original multi-class models.
In this code, I explore this possibility in the context of animal accelerometry, working with example datasets of dogs and seals - dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348)and seals sourced from [Ladds et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166898).
I compare 1-class, binary (1-vs-all), and three kinds of multi-class models on their ability to detect 4 specific behaviours from each of the datasets. For consistency, all models are SVMs.

## Progress Note
In progress. Currently interpretable to me but needs a lot of work to be transferable to others.

## Scripts
There are initialisation scripts that source functions. There is one data exploration RMarkdown file. The remainder of the files are functions.
* [Main Script](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/MainScript.R) <- Load packages, load functions, set experimental variables, split out test data
* [ExploreData](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/ExploreData.Rmd) <- Explore data to determine appropriate target behaviours, window length, etc. When executed, this saved a PDF RMarkdown report with exploratory graphs and tables.
* [Preprocessing](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Preprocessing.R) <- Define behaviour types and window lengths. Currently set to generate timeseries and statistical features. Add other pre-processing such as filtering and normalisation later.
* [HpoOptimisation](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/HpoOptimisation.R) <- Using Bayesian Optimisation search the hyperparameter space and tune models on k-fold cross-validation between training and validation
* [TestBestModels](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/TestBestModels.R) <- Test all OCC and multi-class models on their test sets