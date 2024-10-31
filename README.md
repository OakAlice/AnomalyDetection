# One-Class Classification
Code for a case study demonstrating the use of one-class classification (OCC) for detecting specific target behaviours in animal accelerometer behavioural classification data. Input raw labelled data, select target behaviours and window lengths, generate features and tune model design, testing highest performing design on hold-out set, and calculating final performance for each model and dataset.

Example dataset of 45 dogs sourced from [Vehkaoja et al., 2022](https://www.sciencedirect.com/science/article/pii/S2352340922000348). Dataset of 12 seals sourced from [Ladds et al., 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0166898).

## Disclaimer
The code is currently a bit of a Frankenstein that I've built bit by bit as a I realised I needed more things. It's probably twice as long as it should be. Additionally, it is hyper-specific to the bugs and characteristsics of the data and use-case I developed it for (i.e., demonstrating viability of OCC), so generalising it to other use cases (e.g., doing OCC) may require pulling the code apart. I will do this later, but not yet.

## Scripts
There are initialisation scripts that source functions. There is one data exploration RMarkdown file. The remainder of the files are functions.
* [SetUp](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/SetUp.R) <- Load packages, load functions, set experimental variables, split out test data
* [ExploreData](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/ExploreData.Rmd) <- Explore data to determine appropriate target behaviours, window length, etc. When executed, this saved a PDF RMarkdown report with exploratory graphs and tables.
* [Preprocessing](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/Preprocessing.R) <- Define behaviour types and window lengths. Currently set to generate timeseries and statistical features. Add other pre-processing such as filtering and normalisation later.
* [HpoOptimisation](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/HpoOptimisation.R) <- Using Bayesian Optimisation search the hyperparameter space and tune models on k-fold cross-validation between training and validation
* [TestBestModels](https://github.com/OakAlice/AnomalyDetection/blob/main/Scripts/TestBestModels.R) <- Test all OCC and multi-class models on their test sets

## For my write up
* I added the [output folder](https://github.com/OakAlice/AnomalyDetection/blob/main/Output) that has the final models (rda format) generated for each behaviour as well as a word doc with model performances.