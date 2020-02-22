from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

# Load Test Data
test = pd.read_csv('./test_data.csv', sep=',')
test_labels = pd.read_csv('./test_labels.csv', sep=',')

# Load Model
model = keras.models.load_model('./simple_model')

print("")
loss, mae, mse = model.evaluate(test, test_labels, verbose=2)
print("")
test_mean = np.mean(test_labels.values)
print("Testing set Mean Value: " + str(test_mean))
print("Testing set Mean Abs Error: " + str(mae) + " M/S^2")
percent_error = np.abs(test_mean - mae)/ test_mean
print("Average Percent Error: " + str(percent_error))

test_predictions = model.predict(test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_predictions, test_labels.values)
plt.xlabel('True Values [m/s^2]')
plt.ylabel('Predictions [m/s^2]')
lims = [-5**-3, 5**-4]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()