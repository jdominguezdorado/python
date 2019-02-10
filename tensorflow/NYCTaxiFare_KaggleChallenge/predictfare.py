import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import sys
import argparse

import time
import datetime


# Limits of NYC
NYC_WEST = -74.2795966
NYC_EAST = -73.7001899
NYC_SOUTH = 40.5452258
NYC_NORTH = 40.9147787

def plot_history(history):
	# summarize history for loss
	plt.plot(history.history['mean_absolute_error'])
	plt.plot(history.history['val_mean_absolute_error'])
	plt.title('Model error')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Train error', 'Test error'], loc='upper left')
	plt.show()

def to_timestamp(dt) :
	return time.mktime(datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S %Z").timetuple())

def norm(x):
	return (x - train_stats['mean']) / train_stats['std']	

def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(1)
  ])
  #optimizer = tf.keras.optimizers.RMSprop(0.001)
  optimizer = tf.keras.optimizers.Adadelta()
  model.compile(loss='mae',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def step(message):
  print(' * {}'.format(message))


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

parser = argparse.ArgumentParser(prog='predictfare.py')
parser.add_argument('-e', '--epochs', type=int, default=250, dest='epochs', help='number of iterations to train the network')    
parser.add_argument('-d', '--datafile', default="data\\first1mb.csv", dest='datafile', help='path to the .csv file to read the data from')    
args = parser.parse_args()

step("Reading csv ...\n")
raw_dataset = pd.read_csv(args.datafile,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

step("Copying dataset ...\n")
dataset = raw_dataset.copy()

initial_count = len(dataset.index)

step("Dropping pickup and dropoff points that are not in New York City ...")
dataset.drop(dataset[(dataset.pickup_latitude > NYC_NORTH) | (dataset.pickup_latitude < NYC_SOUTH)].index, inplace=True)
dataset.drop(dataset[(dataset.pickup_longitude > NYC_EAST) | (dataset.pickup_longitude < NYC_WEST)].index, inplace=True)
dataset.drop(dataset[(dataset.dropoff_latitude > NYC_NORTH) | (dataset.dropoff_latitude < NYC_SOUTH)].index, inplace=True)
dataset.drop(dataset[(dataset.dropoff_longitude > NYC_EAST) | (dataset.dropoff_longitude < NYC_WEST)].index, inplace=True)

final_count = len(dataset.index)
print("Removed {} rows\n".format(initial_count-final_count))
initial_count = final_count

step("Pop 'key' column, is the same info that pickup_datetime but in a worse format ...\n")
key = dataset.pop('key')

step("Pop 'passenger_count' column, shouldnt affect the fare amount ...\n")
key = dataset.pop('passenger_count')

step("Converting 'pickup_datetime' from datetime to timestamp ...\n")
dataset['pickup_datetime'] = dataset['pickup_datetime'].apply(to_timestamp)

step("Getting 80%% of the dataset as the train dataset ...\n")
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print('Sample of the train dataset')

train_stats = train_dataset.describe()
train_stats.pop("fare_amount")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('fare_amount')
test_labels = test_dataset.pop('fare_amount')

step('Normalizing train and test data ...')
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print('Sample of the train dataset after the normalization:\n')
print(normed_train_data.tail())

print()
step('Building the model ...')
print('Model summary:')
model = build_model()
model.summary()

print()
step('Training the model to predict the fare amount ...\n')
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history = model.fit(
  normed_train_data, train_labels, 
  epochs=args.epochs, 
  validation_split = 0.2, 
  verbose=0, 
  callbacks=[PrintDot()])

step('\nPlotting training history ...\n')
plot_history(history)

step('Evaluating the network against the test data ...')
metrics = model.evaluate(normed_test_data, test_labels)

print('\nEvaluation metrics:')
for i in range(len(metrics)) : 
	print("{}: {}".format(model.metrics_names[i],metrics[i]))
