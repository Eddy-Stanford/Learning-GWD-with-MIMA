from netCDF4 import Dataset
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import tensorflow as tf

"""
DATA INFO (Name, Shape): 

Longitude (128,)
Longitude Edges (129,)
Latitude (64, )
Latitude Edges (65, )
Pressure (level) (22, )
Time (1440,)
Height (1440, 22, 64, 128)
Level Pressure (slp) (1440, 64, 128)
Zonal Wind Component ucomp (1440, 22, 64, 128)
Meridional Wind Component vcomp (1440,22,64,128)
Dp/dt omega vertical velocity ()
Temperature  (1440,22,64,128)
Gravity wave forcing on mean zonal flow, gwfu_cgwd (1440, 22, 64, 128)
Gravity wave forcing on mean meridional flow, gwfv_cgwd (1440,22,64,128)
"""

"""
DEFINE CONSTANTS

Defines a subarray to use for training & testing
num samples = 10*30*30*30 = 270,000 training & testing examples 
"""
PMAX = 10 # index of max pressure (10 = 10hPa)
PMIN = 0  # index of min pressure (0 = .1hPa)
TMAX = 1440 # index of max time
TMIN = 1430 # index of min time - ONE MONTH
LATMAX = 64 # index of max latitude
LATMIN = 54 # index of min latitude - 
LONMAX = 128 # index of max longitude
LONMIN = 118 # index of min longitude 
NUM_SAMPLES = (TMAX-TMIN)*(PMAX-PMIN)*(LATMAX-LATMIN)*(LONMAX-LONMIN)

DEPTH = 3 # Number of neighbors to include in each sample (naive way to incorporate spatial and temporal dependence)
SAMPLE_LEN = DEPTH**4*6
TRAIN_SPLIT = .8 # 80% of samples used for training

class Data(object): 
    def __init__(self, fileName, verbose=True):
        if verbose: print("Loading Data")
        self.fileName = fileName
        self.cdfData = Dataset(self.fileName, "r", format="NETCDF4")

        # Extra Features
        # self.lat = np.array(self.cdfData.variables['lat'])
        # self.lon = np.array(self.cdfData.variables['lon'])
        # self.time = np.array(self.cdfData.variables['time'])
        # self.level_pressure = np.array(self.cdfData.variables['slp'])

        # Normalized Features 
        # if verbose: print("Normalizing Data")
        self.temp = self.normalize(np.array(self.cdfData.variables['temp'])[:,PMIN:PMAX,:,:])
        self.height = self.normalize(np.array(self.cdfData.variables['hght'])[:,PMIN:PMAX,:,:])
        self.ucomp = self.normalize(np.array(self.cdfData.variables['ucomp'])[:,PMIN:PMAX,:,:])
        self.vcomp = self.normalize(np.array(self.cdfData.variables['vcomp'])[:,PMIN:PMAX,:,:])
        self.omega = self.normalize(np.array(self.cdfData.variables['omega'])[:,PMIN:PMAX,:,:])
        self.pressure = self.normalize(np.array(self.cdfData.variables['level']))

        # Ground Truth Labels
        self.gwfu_cgwd = np.array(self.cdfData.variables['gwfu_cgwd'])
        # self.gwfv_cgwd = np.array(self.cdfData.variables['gwfv_cgwd'])

        # Free memory
        self.cdfData = 0

    def normalize(self, arr):
        amin = arr.min(keepdims=True)
        amax = arr.max(keepdims=True)
        arr = (arr - amin) / (amax - amin)
        return arr
    
    def get_info(self, var):
        self.cdfData = Dataset(fileName, "r", format="NETCDF4")
        return self.cdfData.variables[var]

class DataProcessor(object):
    def __init__(self, rawdata, normalize=False, verbose=True):
        self.verbose = verbose
        self.rawdata = rawdata

        if normalize: 
            if verbose: print("Normalizing Data")
            self.rawdata = self.normalize(rawdata)
            
        if verbose: print("Collecting Samples")
        self.data, self.labels = self.collectSamples()
        print(self.data.shape)

        self.rawdata = 0 # Remove from memory

        if verbose: print("Splitting Data")
        self.train, self.test = np.split(self.data, [int(TRAIN_SPLIT*NUM_SAMPLES)], axis=0)
        self.train_labels, self.test_labels = np.split(self.labels, [int(TRAIN_SPLIT*NUM_SAMPLES)], axis=0)
        
        if verbose: 
            print("Train: ", self.train.shape)
            print("Train Labels: ", self.train_labels.shape)
            print("Test: ", self.test.shape)
            print("Test Labels: ",  self.test_labels.shape)

    def collectSamples(self): 
        data = np.empty([NUM_SAMPLES, SAMPLE_LEN])
        labels = np.empty([NUM_SAMPLES, 1])
        for t in range(TMIN, TMAX):
            for p in range(PMIN, PMAX):
                for lat in range(LATMIN, LATMAX):
                    for lon in range(LONMIN, LONMAX):
                        idx = np.ravel_multi_index([t-TMIN, p-PMIN, lat-LATMIN, lon-LONMIN], (TMAX-TMIN, PMAX-PMIN, LATMAX-LATMIN, LONMAX-LONMIN))
                        data[idx] = self.createSample(t,p,lat,lon)
                        labels[idx] = (10**5)*self.rawdata.gwfu_cgwd[t][p][lat][lon]
        return data, labels

    def createSample(self, t, p, lat, lon):
        sample = [] 
        for loD in range(DEPTH):
            for laD in range(DEPTH):
                for pD in range(DEPTH): 
                    for tD in range(DEPTH): 
                        temp = self.rawdata.temp[t-tD][p-pD][lat-laD][lon-loD]
                        height = self.rawdata.height[t-tD][p-pD][lat-laD][lon-loD]
                        ucomp = self.rawdata.ucomp[t-tD][p-pD][lat-laD][lon-loD]
                        vcomp = self.rawdata.vcomp[t-tD][p-pD][lat-laD][lon-loD]
                        omega = self.rawdata.omega[t-tD][p-pD][lat-laD][lon-loD]
                        pressure = self.rawdata.pressure[p]
                        sample = np.append(sample,[temp,height,ucomp,vcomp,omega,pressure])
        return np.array(sample)

fileName = "../atmos_1day_d11160_plevel.nc"
rawdata = Data(fileName)
dataset = DataProcessor(rawdata)

def build_model():
  model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[dataset.train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

early_history = model.fit(dataset.train, dataset.train_labels, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop])

print(early_history.history.keys())
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(early_history.history['val_loss'], label='val_loss')
ax.plot(early_history.history['mae'], label='mae')
ax.plot(early_history.history['mse'], label='mse')
plt.xlabel('Epochs')
ax.legend()
plt.show()

loss, mae, mse = model.evaluate(dataset.test, dataset.test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} m/s^2".format(mae))

test_predictions = model.predict(dataset.test).flatten()

a = plt.axes(aspect='equal')
plt.scatter(dataset.test_labels, test_predictions)
plt.xlabel('True Values [m/s^2]')
plt.ylabel('Predictions [m/s^2]')
lims = [-50, 10]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.show()
