# Decomposing Complex Problems
Complex classification problems can be decomposed into a series of easier classification problems allowing us to target specific classes and make fewer assumptions about uncertain classes. In theory, an ensemble of these simpler models should perform as well as, or better than, the original multi-class models for open set recongition problems (where there are classes in real set not in training set).
In this code, I explore this possibility in the context of animal accelerometry, working with example datasets of dogs (sourced from [Ferdinandy et al., 2020](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236092)).
I compare multiple kinds of SVMs, 1-class, binary (1-vs-all), and three kinds of multi-class models on their ability to detect specific behaviours from a common test set when trained on progressively fewer classes.

## Progress Note
In progress.

## Scripts
### R
There are initialisation scripts that source functions. There is one data exploration RMarkdown file. The remainder of the files are functions.
* [MainScript,R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/MainScript.R) <- Load packages, load functions, set experimental variables, split out test data
* [Preprocessing.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/Preprocessing.R) <- Define behaviour types and window lengths. Currently set to generate timeseries and statistical features. Add other pre-processing such as filtering and normalisation later.
* [StatsMarkdownR.R](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/R/StatsMarkdown.Rmd) <- Generate stats and plots for the comparative performance of each of the models

### Python
Eventually I switched my workflow over to python to utilise some of the ML libraries available there.
* [requirements.txt](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/requirements.txt) <- packages
* [MainScript.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/MainScript.py) <- Define variables
* [CreateDatasets.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/CreateDatasets.py) <- split out data into subframes based on TRAINING_SET, clean data, save to disk for easier retrieval later
* [HpoOptimisation.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/HpoOptimisation.py) <- Bayesian search hyperparameters to identify optimal model variables
* [TrainModel.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/TrainModel.py) <- Train optimal model based on identified optimal hyperparameters
* [TestModelOpen.py](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Python/TestModelOpen.py) <- Test model on complete test set

