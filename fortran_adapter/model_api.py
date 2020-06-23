import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def predict(*args):
    print("STARTING PREDICTION")
    arr = np.array(args[0])
    print(arr)
    verbose = args[1]

    if verbose: print("ARRAY: ", arr.shape)
    # Load Model
    if verbose: print("LOADING MODEL")
    model = keras.models.load_model('../simple_model')
    # Test Model
    if verbose: print("PREDICTING")
    test_prediction = model.predict(arr)[0][0]

    if verbose: print("PYTHON PREDICTION: ", test_prediction)

    return test_prediction

# arr = np.ones((1,486))
# predict(arr, True)
