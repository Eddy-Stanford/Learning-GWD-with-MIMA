from netCDF4 import Dataset
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

class DataModel(object):
  def __init__(self, fileNames):
    self.train = pd.read_csv(fileNames['train'], sep=',')
    self.train_labels = pd.read_csv(fileNames['train_labels'], sep=',')
    self.test = pd.read_csv(fileNames['test'], sep=',')
    self.test_labels = pd.read_csv(fileNames['test_labels'], sep=',')

class SimpleModel(): 
  def __init__(self, input_shape, save_path, verbose=True):
    '''
    Defines a simple keras neural network. 

    :param Int input_shape: shape of input training data
    :param Str save_path: Path to save network during training
    :param Bool verbose: True to print network details
    '''
    self.model = keras.Sequential([
      layers.Dense(128, activation='relu', input_shape=[input_shape]),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    self.model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    # Define Callbacks
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0)
    # Saves model during and at the end of training
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=save_path, verbose=1, period=10)
    self.callbacks = [early_stop, checkpoint]

    if verbose: self.model.summary()

  def train(self, train_data, train_labels, epochs):
    self.early_history = self.model.fit(train_data, train_labels, 
                        epochs=epochs, validation_split = 0.2, verbose=0, 
                        callbacks=self.callbacks)
    return self.early_history

  def plot_metrics(self):
    """
    Plots mae, mse, and val_loss while training. Consider moving
    outside the SimpleModel class, so that it can be shared between models 
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(self.early_history.history['val_loss'], label='val_loss')
    ax.plot(self.early_history.history['mae'], label='mae')
    ax.plot(self.early_history.history['mse'], label='mse')
    plt.xlabel('Epochs')
    ax.legend()
    plt.show()

  def evaluate(self, test, test_labels):
      loss, mae, mse = self.model.evaluate(test, test_labels, verbose=2)
      print("Testing set Mean Abs Error: " + str(mae) + "M/S^2")


fileNames = {
  'train': './data/train_data.csv',
  'train_labels': './data/train_labels.csv',
  'test': './data/test_data.csv',
  'test_labels': './data/test_labels.csv',
}

epochs = 1000

data = DataModel(fileNames)
# Instantiate Model
model = SimpleModel(input_shape=data.train.shape[1], save_path='./simple_model')
# Train Model
model.train(data.train, data.train_labels, epochs)
# Plot Training Metrics
model.plot_metrics()
# Gives initial evaluation of network
model.evaluate(data.test, data.test_labels)