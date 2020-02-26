from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

# Load Test Data
test = pd.read_csv('./test_data.csv', sep=',')
test_labels = pd.read_csv('./test_labels.csv', sep=',')
# Load Train Data
train = pd.read_csv('./train_data.csv', sep=',')
train_labels = pd.read_csv('./train_labels.csv', sep=',')

# Load Model
model = keras.models.load_model('./simple_model')

# These values are scaled up due to initial data processing: 
# either scale down or ignore
# loss, mae, mse = model.evaluate(test, test_labels, verbose=2)

# Test Model
test_predictions = model.predict(test)

# Correct values by scaling down
test_predictions = test_predictions*(10**-7)
test_labels = test_labels*(10**-7)
test_predictions = np.reshape(np.array(test_predictions), test_predictions.shape[0],1)
test_labels = np.array(test_labels)

# Evaluate Model 
mae = np.mean(np.absolute(np.subtract(test_labels, test_predictions)))
print("-----------------------------")
print("Mean Absolute Error: ", mae)
print("-----------------------------")

def linear_p_vs_t():
    """
    Creates 2 plots
    Plot 1: 
        Red: num test labels vs test labels
        Blue: num prediction labels vs prediction labels 
    Plot 2: 
        num labels vs difference between test and prediction labels
    """
    nlabels = test_labels.shape[0]
    fig, ax = plt.subplots()
    ax.plot(test_labels, range(nlabels), 'r', label='truth')
    ax.plot(test_predictions, range(nlabels), 'b', label='predictions')
    legend = ax.legend(loc='upper center')
    plt.xlabel('gwdv m/s^2')
    plt.ylabel('Sample')
    plt.show()

def prediction_vs_truth():
    """
    Creates 1 plot
    x axis = true values
    y axis = predicted values
    """
    a = plt.axes(aspect='equal')
    plt.scatter(test_predictions, test_labels)
    plt.xlabel('True Values [m/s^2]')
    plt.ylabel('Predictions [m/s^2]')
    lims = [-5e-4, 4e-5]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    plt.show()

linear_p_vs_t()
prediction_vs_truth()