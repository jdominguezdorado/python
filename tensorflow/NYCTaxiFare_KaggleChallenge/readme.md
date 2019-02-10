
# Kaggle New York City Taxi Fare Prediction challenge

This is my take on the machine learning challenge published at [Kaggle Page](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction). The test file included in the data/ subdirectory is only the first 1mb of the more than 5GB of statistical data that the Kaggle page supplies for the challenge. You can download the rest from [there](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data).

## Requirements

To run it you will need to follow the following steps:

 - Install Python 3.6 with pip (no TensorFlow binary can be installed via pip for later versions. If you have a later version already installed and don't want to uninstall it, try virtualenv)
 - pip install tensorflow
 - pip install pandas
 - pip install matplotlib

## Command line arguments

All the command line arguments have default values, so you can run the script just with ./predictfare.py

Current available arguments include:

  - -e EPOCHS, --epochs EPOCHS
                        number of iterations to train the network
  - -d DATAFILE, --datafile DATAFILE
                        path to the .csv file to read the data from

In the future you'll be able to change additional parameters such as the number of layers, number of neurons or the optimization function.
