import csv
import os
from collections import defaultdict
from random import randint, seed
from typing import Dict, Union

import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking
from lrgwd.extractor.config import DEFAULTS
from lrgwd.extractor.utils import extract_3D_tensors, extract_tensors, Data


@click.command("extractor")
@click.option(
    "--save-path",
    default=DEFAULTS["save_path"],
    show_default=True,
    type=str,
    help="File path to save extracted dataset",
)
@click.option(
    "--source-path",
    default=DEFAULTS["source_path"],
    show_default=True,
    type=str,
    help="File path to raw dataset as npz",
)
@click.option(
    "--correlations/--no-correlations",
    default=True,
    show_default=True,
    help="Include correlations and plots"
)
@click.option(
    "--plevels-included",
    default=DEFAULTS["plevels_included"],
    type=int,
    help="Use only the top N plevels"
)
@click.option(
    "--num-samples",
    default=DEFAULTS["num_samples"],
    show_default=True,
    type=int,
    help="Num of samples to creates. Defaults to using all data in source path"
)
@click.option(
    "--generate-cnn-features/--no-generate-cnn-features", 
    default=False,
    show_default=True,
    help="Generate 3D features matricies"
)
@click.option(
    "--step-size",
    default=DEFAULTS["CNN_features"]["step_size"],
    type=int,
    help="Step size in 6hr increments between vertical columns in 3d features"
)
@click.option(
    "--start-time",
    default=DEFAULTS["CNN_features"]["start_time"],
    type=int,
    help="Number of 6hr increments to before current sample to start when creating 3d features"
)
@click.option(
    "--num-steps",
    default=DEFAULTS["CNN_features"]["num_steps"],
    type=int,
    help="Number of steps to go back in time from current vertical columns when constructing 3d features"
)
@click.option(
    "--tracking/--no-tracking",
    default=True,
    show_default=True,
    help="Track run using mlflow"
)
@click.option(
    "--batch-size",
    default=DEFAULTS["batch_size"],
    show_default=True,
    type=int,
    help="Number of feature vectors to process before writing"
)
@click.option("--verbose/--no-verbose", default=True)
def main(**params):
    """
    Extracts samples from raw dataset.
    """
    with tracking(
        experiment="extractor",
        params=params,
        local_dir=params["save_path"],
        tracking=params["tracking"]
    ):
        with np.load(params["source_path"], allow_pickle=False) as npz_data:
            # Memory Load data
            if params["verbose"]:
                logger.info(f"Memory Loading Data")

            data = Data(dict(npz_data))
            os.makedirs(params["save_path"], exist_ok=True)

            # Create FEATURE TENSORS
            if params["verbose"]:
                logger.info(f"Generate Feature Tensors")

            if params["generate_cnn_features"]:
                extract_3D_tensors(
                    data=data,
                    save_path=params["save_path"],
                    step_size=params["step_size"],
                    num_steps=params["num_steps"],
                    start_time=params["start_time"],
                    num_samples=params["num_samples"],
                )
            else: 
                extract_tensors(
                    data=data,
                    save_path=params["save_path"],
                    num_samples=params["num_samples"],
                    plevels=params["plevels_included"],
                    batch_size=params["batch_size"]
                )


if __name__ == "__main__":
    main()





# if params["correlations"]:
#     logging.info(f"Find and plot spearman correlation")
#     generate_temporal_correlation_plots(
#         data=data,V 
#         save_path=params["extracted_data_path"]
#     )
#     generate_spatial_correlation_plots(
#         data=data, 
#         save_path=params["extracted_data_path"]
#     )
# def generate_feature_vectors(
#     data: Dict[str, np.ndarray], 
#     save_path: Union[os.PathLike, str],
# ) -> None:
#     """
#     Create feature vectors and labels.  
#     """
#     fvectors_path = os.path.join(save_path, "feature_vectors.csv")
#     gwfu_labels_path = os.path.join(save_path, "gwfu_labels.csv")
#     gwfv_labels_path = os.path.join(save_path, "gwfv_labels.csv")

#     _extract_feature_vectors(
#         data=data,
#         fvectors_file=fvectors_path, 
#         gwfu_labels_file=gwfu_labels_path,
#         gwfv_labels_file=gwfv_labels_path,
#     )

# def _extract_feature_vectors(
#     data: Dict[str, np.ndarray],
#     fvectors_file: str,  
#     gwfu_labels_file: str, 
#     gwfv_labels_file: str,
# ) -> None:
#     """
#     Number of feats: 5*7*3 + 3*3 = 114
#     """
#     # Add 1 because for depth=D, 1 on them will be 0 index
#     DIMS = (
#         1440,
#         22 - SPATIAL_DEPTH, 
#         64 - SPATIAL_DEPTH,
#         128 - SPATIAL_DEPTH,
#     )

#     # The start index is equal to the border that is being removed. It must be removed from both the
#     # start and end of the dataframe.
#     start_indx = np.ravel_multi_index(
#         multi_index=([TIME_DEPTH, SPATIAL_DEPTH, SPATIAL_DEPTH, SPATIAL_DEPTH]), 
#         dims=DIMS
#     )
#     end_indx = np.array(DIMS).prod() - start_indx

#     # remove fvectors that do not have neighbors (i.e. borders)
#     # NUM_FVECTORS = np.array(DIMS).prod() - start_indx
#     NUM_FVECTORS = 20 + start_indx

#     logging.info("Extracting Vectors")

#     # Initialize empty dataframe
#     fvectors = pd.DataFrame()
#     for i in tqdm(range(start_indx, NUM_FVECTORS), "feature vectors"):
#         try:
#             t, plevel, lat, lon = np.unravel_index(i, DIMS)
#         except ValueError: 
#             logging.error(f"Index {i} is out of bounds. Exiting now.")
#             return

#         fvector = pd.DataFrame()
#         for tD in range(TIME_DEPTH): 
#             for sD in range(-(SPATIAL_DEPTH), (SPATIAL_DEPTH + 1)): # Using spatial depth: range(-1, 2) == [-1, 0, 1]
#                 # Add pressure level
#                 plevel_value = pd.DataFrame(
#                     data=[data["level"][plevel + sD]], 
#                     columns=[f"plevel"]
#                 )
#                 fvector = pd.concat([fvector, plevel_value], axis=1, copy=False)

#                 if sD == 0: 
#                     for feat_label in FEATURES:
#                         feat_value = pd.DataFrame(
#                             data=[data[feat_label][t - tD, plevel, lat, lon]],
#                             columns=[f"{feat_label}_t{tD}_x0_y0_z0"]
#                         )
#                         fvector = pd.concat([fvector, feat_value], axis=1, copy=False)
#                     continue

#                 # Z (plevel):
#                 for feat_label in FEATURES:
#                     feat_value = pd.DataFrame(
#                         data=[data[feat_label][t - tD, plevel + sD, lat, lon]],
#                         columns=[f"{feat_label}_t{tD}_x0_y0_z{sD}"]
#                     )
#                     fvector = pd.concat([fvector, feat_value], axis=1, copy=False)
                
#                 # Y (lon):
#                 for feat_label in FEATURES:
#                     feat_value = pd.DataFrame(
#                         data=[data[feat_label][t - tD, plevel, lat, lon + sD]],
#                         columns=[f"{feat_label}_t{tD}_x{0}_y{sD}_z0"]
#                     )
#                     fvector = pd.concat([fvector, feat_value], axis=1, copy=False)

#                 # X (lat):
#                 for feat_label in FEATURES:
#                     feat_value = pd.DataFrame(
#                         data=[data[feat_label][t - tD, plevel, lat + sD, lon]],
#                         columns=[f"{feat_label}_t{tD}_x{sD}_y0_z0"]
#                     )
#                     fvector = pd.concat([fvector, feat_value], axis=1, copy=False)
        
#         fvectors = pd.concat([fvectors, fvector])
#         print(fvectors.shape)
#         print(fvectors)
#         # Only include header on first batch
#         include_header = False
#         if (i - start_indx) == (BATCH_SIZE-1): include_header = True 
#         # write batch of fvectors and labels
#         if fvectors.shape[0] == BATCH_SIZE:

#             # Write Feature Vectors
#             fvectors.to_csv(fvectors_file, mode='a', header=include_header, index=False)

#             # Get batch window
#             t_prev, plevel_prev, lat_prev, lon_prev = np.unravel_index(i - BATCH_SIZE, DIMS)

#             if t_prev == t: t_prev -= 1
#             if plevel_prev == plevel: plevel_prev -= 1
#             if lat_prev == lat: lat_prev -= 1 
#             if lon_prev == lon: lon_prev -= 1 

#             # Write Labels
#             gwfu_batch_labels = data["gwfu_cgwd"][t_prev:t, plevel_prev:plevel, lat_prev:lat, lon_prev:lon]
#             gwfu_batch_labels = gwfu_batch_labels.flatten()
#             gwfu_df = pd.DataFrame(gwfu_batch_labels)
#             gwfu_df.to_csv(gwfu_labels_file, mode='a', header=include_header, index=False) 

#             gwfv_batch_labels = data["gwfv_cgwd"][t_prev:t, plevel_prev:plevel, lat_prev:lat, lon_prev:lon]
#             gwfv_batch_labels = gwfv_batch_labels.flatten()
#             gwfv_df = pd.DataFrame(gwfv_batch_labels)
#             gwfv_df.to_csv(gwfv_labels_file, mode='a', header=include_header, index=False) 


#             # clear batch
#             fvectors = pd.DataFrame()
                

# def generate_temporal_correlation_plots(
#     data: Dict[str, np.ndarray],
#     save_path: Union[os.PathLike, str],
# ) -> None:
#     """
#     Calculates correlations between gwd at the hard coded time for all positions with
#     each feature in FEATURES. Generates one plot to understand how  correlation varies
#     with change in time.

#     Parameters: 
#     -----------
#         npz_data : Memory mapped raw numpy data
#         save_path : Path to save generated plots and correlations

#     Returns: 
#     --------
#         None

#     """
#     t, plevel, lon = (0,3, 30)
#     gwfu = data[TARGET[0]][t, plevel, :, :].flatten()
    
#     trange = np.arange(0, 500)

#     correlations = defaultdict(lambda: [])
#     for sub_t in tqdm(trange, "time steps"):
#         for feature in FEATURES:
#             feat = data[feature][sub_t, plevel, :, :].flatten()
#             rho, _ = pearsonr(feat, gwfu)
#             correlations[feature].append(rho)
    
#     for feature in FEATURES:
#         plt.plot(trange, correlations[feature], label=f"{feature}")
    
#     plt.title(f"GWFU rho at {t}, {plevel} varying time") 
#     plt.xlabel("time steps")
#     plt.ylabel("pearsonr correlation (-1,1)")
#     plt.legend()

#     correlations_path = os.path.join(save_path, "temporal_correlations.png")
#     plt.savefig(correlations_path)
#     plt.show()


# def generate_spatial_correlation_plots(
#     data: Dict[str, np.ndarray], 
#     save_path: Union[os.PathLike, str],
# ) -> None:
#     """
#     Calculates correlations between gwd at the hard coded position for all time with
#     each feature in FEATURES at a position that varies with lat and lon for all time. Generates
#     two plots to understand how correlation varies with change in latitude and longitude.  

#     Parameters: 
#     -----------
#         npz_data : Memory mapped raw numpy data
#         save_path : Path to save generated plots and correlations

#     Returns: 
#     --------
#         None

#     """
#     plevel, lat, lon = (3, 30, 60)
#     gwfu = data[TARGET[0]][:, plevel, lat, lon].flatten()

#     latrange = np.arange(0, 50)
#     lonrange = np.arange(30,80)

#     # Generate Correlations
#     correlations = defaultdict(lambda: defaultdict(lambda: []))
#     for sub_lat, sub_lon in tqdm(zip(latrange, lonrange), "lat/lon range"):
#         for feature in FEATURES: 
#             # Lat 
#             feat = data[feature][:, plevel, sub_lat, lon].flatten()
#             rho, _ = spearmanr(feat, gwfu)
#             correlations["lat"][feature].append(rho)
#             # Lon
#             feat = data[feature][:, plevel, lat, sub_lon].flatten()
#             rho, _ = spearmanr(feat, gwfu)
#             correlations["lon"][feature].append(rho)

#     # Plot Correlations
#     fig, axs = plt.subplots(1,2)
#     for feature in FEATURES:
#         # LAT
#         axs[0].plot(latrange, correlations["lat"][feature], label=f"{feature}") 
#         # LON
#         axs[1].plot(lonrange, correlations["lon"][feature], label=f"{feature}") 

#     axs[0].set_title(f"GWFU rho at {plevel}, {lat}, {lon} varying lat")
#     axs[0].legend()
#     axs[1].set_title(f"GWFU rho at {plevel}, {lat}, {lon} varying lon")
#     axs[1].legend()

#     correlations_path = os.path.join(save_path, "spatial_correlations.png")
#     fig.savefig(correlations_path)
#     plt.show()

# def scatter_plot_full_features(
#     data: Dict[str, np.ndarray], 
#     save_path: os.PathLike, 
#     num_points: int
# ) -> None:
#     """
#     This function creates a scatter plot for each feature.
#     Helps to visualize some of the correlations
#     """
#     fig, axs = plt.subplots(2, 3)

#     gwfu = data[TARGET[0]].flatten()
#     idx = np.random.choice(np.arange(len(gwfu)), num_points, replace=False)
#     gwfu = gwfu[idx]
#     nonzero_gwfu = np.nonzero(gwfu)
#     gwfu = gwfu[nonzero_gwfu]

#     xmin, xmax = (np.amin(gwfu), np.amax(gwfu))
    
#     # Generate Color Map
#     colors_idx = np.floor(idx / (1440*64*128))
#     colors = cm.RdYlBu(np.linspace(0,1,22))
#     colors = np.array([colors[int(x)] for x in colors_idx])
#     colors = colors[nonzero_gwfu]


#     for i, feat_name in enumerate(FEATURES):
#         logging.info(f"{feat_name}: Scatter Plot and Correlation")
#         feat = data[FEATURES[i]].flatten()[idx]
#         feat = feat[nonzero_gwfu] 

#         xindx, yindx = np.unravel_index(i, (2,3))

#         axs[xindx][yindx].scatter(gwfu, feat, c=colors)
#         axs[xindx][yindx].set_xlim(xmin, xmax)
#         axs[xindx][yindx].xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
#         axs[xindx][yindx].set_ylabel(f"{feat_name}")            

#     plt.show()

# ####### CODE FOR SCATTER PLOTS ########
# # def visualize_correlations(npz_data):
# #     fig, axs = plt.subplots(2, 3)
# #     gwfu = npz_data[TARGET[0]]

# #     seed(42)
# #     lat = randint(0, 64)
# #     lon = randint(0, 128)
# #     plevel = 5

# #     gwfu = gwfu[:,plevel,lat,lon].flatten()
# #     xmin, xmax = (np.amin(gwfu), np.amax(gwfu))

# #     # Temperate vs GWFU
# #     temp = npz_data["temp"][:, plevel, lat, lon].flatten()
# #     axs[0][0].set_xlim(xmin, xmax)
# #     axs[0][0].scatter(gwfu, temp) 
    
# #     # Ucomp vs GWFU
# #     ucomp = npz_data["ucomp"][:, plevel, lat, lon].flatten()
# #     axs[0][1].set_xlim(xmin, xmax)
# #     axs[0][1].scatter(gwfu, ucomp) 

# #     # Vcomp vs GWFU
# #     vcomp = npz_data["vcomp"][:, plevel, lat, lon].flatten()
# #     axs[1][0].set_xlim(xmin, xmax)
# #     axs[1][0].scatter(gwfu, vcomp) 

# #     # Omega vs GWFU
# #     omega = npz_data["omega"][:, plevel, lat, lon].flatten()
# #     axs[1][1].set_xlim(xmin, xmax)
# #     axs[1][1].scatter(gwfu, omega) 

# #     # Hght vs GWFU
# #     hght = npz_data["hght"][:, plevel, lat, lon].flatten()
# #     axs[1][2].set_xlim(xmin, xmax)
# #     axs[1][2].scatter(gwfu, hght) 
# #     plt.show()
# #     return

    
# # logger.info(f"Finding Aggregate Spearman's RC")

# #     fig, axs = plt.subplots(2, 3)
# #     gwfu = npz_data[TARGET[0]]

# #     seed(42)
# #     lat = randint(0, 64)
# #     lon = randint(0, 128)
# #     plevel = 5

# #     gwfu = gwfu[:,plevel,lat,lon].flatten()
# #     xmin, xmax = (np.amin(gwfu), np.amax(gwfu))

# #     # Temperate vs GWFU
# #     temp = npz_data["temp"][:, plevel, lat, lon].flatten()
# #     axs[0][0].set_xlim(xmin, xmax)
# #     axs[0][0].scatter(gwfu, temp) 
    
# #     # Ucomp vs GWFU
# #     ucomp = npz_data["ucomp"][:, plevel, lat, lon].flatten()
# #     axs[0][1].set_xlim(xmin, xmax)
# #     axs[0][1].scatter(gwfu, ucomp) 

# #     # Vcomp vs GWFU
# #     vcomp = npz_data["ucomp"][:, plevel, lat, lon].flatten()
# #     axs[1][0].set_xlim(xmin, xmax)
# #     axs[1][0].scatter(gwfu, vcomp) 

# #     plt.show()
