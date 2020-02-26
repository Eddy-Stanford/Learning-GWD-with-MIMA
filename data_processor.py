from netCDF4 import Dataset
from scipy import stats
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

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
SCALING_FACTOR = 10**7

class RawData(object): 
    def __init__(self, fileName, verbose=True):
        """
        Imports raw CDF data from fileName and extracts variables in relevant window
        """
        VARIABLES = ['temp', 'hght', 'ucomp', 'vcomp', 'omega', 'level', 'gwfu_cgwd']

        if verbose: print("Load Data")
        self.fileName = fileName
        self.cdfData = Dataset(self.fileName, "r", format="NETCDF4")
        self.cdfData = dict((key, self.cdfData.variables[key]) for key in VARIABLES)

        if verbose: print("Extract Variables")
        self.temp = np.array(self.cdfData['temp'])[:,PMIN:PMAX,:,:]
        self.height = np.array(self.cdfData['hght'])[:,PMIN:PMAX,:,:]
        self.ucomp = np.array(self.cdfData['ucomp'])[:,PMIN:PMAX,:,:]
        self.vcomp = np.array(self.cdfData['vcomp'])[:,PMIN:PMAX,:,:]
        self.omega = np.array(self.cdfData['omega'])[:,PMIN:PMAX,:,:]
        self.pressure = np.array(self.cdfData['level'])
        # Ground Truth Labels
        self.gwfu_cgwd = np.array(self.cdfData['gwfu_cgwd'])[:,PMIN:PMAX,:,:]
        # self.gwfv_cgwd = np.array(self.cdfData.variables['gwfv_cgwd'])

        if verbose: 
            print("Describe RawData")
            print("----------------")
            print('(time, level, lat, lon): ', np.array(self.cdfData['temp']).shape)    
            temp_stats = stats.describe(self.temp, axis=None)
            height_stats = stats.describe(self.height, axis=None)
            ucomp_stats = stats.describe(self.ucomp, axis=None)
            vcomp_stats = stats.describe(self.vcomp, axis=None)
            omega_stats = stats.describe(self.omega, axis=None)
            gwfu_cgwd_stats = stats.describe(self.gwfu_cgwd, axis=None)
            df = pd.DataFrame([temp_stats, height_stats, ucomp_stats, vcomp_stats, omega_stats, gwfu_cgwd_stats],
                        columns=['num obs', 'minmax', 'mean', 'var', 'skew', 'kurtosis'],
                        index=['temp (deg_k)', 'height (m)', 'ucomp (m/s)', 'vcomp (m/s)', 'omega (Pa/s)', 'gwfv_cgwd (m/s^2)'])

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)

            print('Pressure Levels: ', self.pressure)
            
class DataProcessor(object):
    def __init__(self, rawdata, save=True, verbose=True):
        '''
        Dataprocess 
        1. Imports rawdata, 
        2. Collects samples
        3. Splits samples into training and test data,
        4. Standardizes data
        5. Saves data to csv files
        '''
        self.rawdata = rawdata

        if verbose: print("Collect Samples")
        self.data, self.labels = self.collectSamples()
        self.labels = self.labels*SCALING_FACTOR

        if verbose: print("Split Data")
        self.train, self.test = np.split(self.data, [int(TRAIN_SPLIT*NUM_SAMPLES)], axis=0)
        self.train_labels, self.test_labels = np.split(self.labels, [int(TRAIN_SPLIT*NUM_SAMPLES)], axis=0)

        if verbose: print("Standardize")
        scaler_features = StandardScaler()
        self.train = scaler_features.fit_transform(self.train)
        self.test = scaler_features.transform(self.test)

        if save:
            if verbose: print("Save Processed Data")
            comment = "First 10 pressure levels. Standardized data across features. " + "Split: " + str(TRAIN_SPLIT) + "Depth: " + str(DEPTH)
            np.savetxt('./data/train_data.csv', self.train, delimiter=',', comments=comment)
            np.savetxt('./data/train_labels.csv', self.train_labels, delimiter=',')

            np.savetxt('./data/test_data.csv', self.test, delimiter=',')
            np.savetxt('./data/test_labels.csv', self.test_labels, delimiter=',')

    def collectSamples(self): 
        '''
        Returns data samples and corresponding training labels. 
        Each sample contains temp, height, ucomp, vcomp, omega, and pressure at a 
        lat,lon, pressure level, and time step, as well as those values for relevant neighbors. 
        Each label contains a gwd value at the corresponding lat,lon, pressure level and time step. 
        '''
        data = np.empty([NUM_SAMPLES, SAMPLE_LEN])
        labels = np.empty([NUM_SAMPLES, 1])
        for t in range(TMIN, TMAX):
            for p in range(PMIN, PMAX):
                for lat in range(LATMIN, LATMAX):
                    for lon in range(LONMIN, LONMAX):
                        idx = np.ravel_multi_index([t-TMIN, p-PMIN, lat-LATMIN, lon-LONMIN], (TMAX-TMIN, PMAX-PMIN, LATMAX-LATMIN, LONMAX-LONMIN))
                        data[idx] = self.createSample(t,p,lat,lon)
                        labels[idx] = self.rawdata.gwfu_cgwd[t][p][lat][lon]
        return data, labels

    def createSample(self, t, p, lat, lon):
        '''
        Returns a single training sample. 
        Includes nearest neighbors to depth determined by hyperparam
        
        :param t time: time step sample is centered on
        :param p pressure level: pressure level sample is centered on
        :param lat latitude: latitude sample is centered on
        :param lon longitude: longitude sample is centered on
        '''
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

"""
RUN DATAMODEL
"""
fileName = "../../atmos_1day_d11160_plevel.nc"
rawdata = RawData(fileName, verbose=True)
dataset = DataProcessor(rawdata, save=True)

