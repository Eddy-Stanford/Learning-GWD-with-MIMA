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

MIMA = {
    "gwfu": {
        "time": 1440,
        "plevel": 22,
        "lat": 64,
        "lon": 128
    }
}

FEATURES = ["temp", "hght", "ucomp", "vcomp", "omega", "level", "slp", "gwfu_cgwd", "gwfv_cgwd"]

# TOP 18 pressure levels have non zero gwd values
NON_ZERO_GWD_PLEVELS = 18

TENSOR = ["temp", "hght", "ucomp", "vcomp", "omega", "slp", "gwfu_cgwd", "gwfv_cgwd"]
VERTICAL_COLUMN_FEATURES = ["slp", "lat", "lon"]

TRAIN_FEATURES = ["temp", "hght", "ucomp", "vcomp", "omega", "slp", "lat", "lon"]
TARGET_FEATURES = ["gwfu_cgwd", "gwfv_cgwd"]

# FEATURE_INFO = {
#     "time": {
#         "units": "6 hr intervals",
#         "long_name": "time",
#     },
#     "temp": {
#         "units": "deg_k",
#         "long_name": "temperature",
#         "valid_range": [100., 400.],
#         "shape": (1440, 22, 64, 128),
#     },
#     "hght": {
#         "units": 
#         "long_name":
#     },
#     "ucomp": {
#         "units":
#         "long_name":
#     },
#     "vcomp": {
#         "units":
#         "long_name":
#     },
#     "omega": {
#         "units":
#         "long_name":
#     },
#     "level": {
#         "units":
#         "long_name":
#     },
#     "slp": {
#         "units":
#         "long_name":
#     },
#     "gwfu_cgwd": {
#         "units":
#         "long_name":
#     },
#     "gwfv_cgwd": {
#         "units":
#         "long_name":
#     },
# }

DEFAULT_CONFIG = {
    "PMAX": 10,  # index of max pressure (10 = 10hPa)
    "PMIN": 3,  # index of min pressure (0 = .1hPa)
    "TMAX": 1440,  # index of max time
    "TMIN": 40,  # index of min time - ONE MONTH
    "LATMAX": 64,  # index of max latitude
    "LATMIN": 63,  # index of min latitude
    "LONMAX": 128,  # index of max longitude
    "LONMIN": 127,  # index of min longitude
    "NUM_SAMPLES": 7
    * 1400
    * 1
    * 1,  # (TMAX-TMIN)*(PMAX-PMIN)*(LATMAX-LATMIN)*(LONMAX-LONMIN)
    "DEPTH": 2,  # Number of neighbors to include in each sample (naive way to incorporate spatial and temporal dependence)
    "SAMPLE_LEN": 2 ** 4 * 6,  # DEPTH**4*6
    "TRAIN_SPLIT": 0.80,  # 80% of samples used for training
    "SCALING_FACTOR": 10 ** 7,
    "MIMA": {
        "FEATURES": ["temp", "hght", "ucomp", "vcomp", "omega", "level", "gwfu_cgwd"]
    },
    "FEATURES": ["temp", "hght", "ucomp", "vcomp", "omega", "level", "gwfu_cgwd"]
}
