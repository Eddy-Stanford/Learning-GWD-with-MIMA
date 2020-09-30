import os
import pickle

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from netCDF4 import Dataset as netcdf_dataset

from lrgwd.utils.io import from_pickle

T = 0
PLEVEL = 25

 # set the colormap and centre the colorbar                                                                                                                                                              
class MidpointNormalize(colors.Normalize):                                                                                                                                                              
    """                                                                                                                                                                                                 
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)                                                                                          

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))                                                                                                                  
    """                                                                                                                                                                                                 
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):                                                                                                                                
        self.midpoint = midpoint                                                                                                                                                                        
        colors.Normalize.__init__(self, vmin, vmax, clip)                                                                                                                                               

    def __call__(self, value, clip=None):                                                                                                                                                               
        # I'm ignoring masked values and all kinds of edge cases to make a                                                                                                                              
        # simple example...                                                                                                                                                                             
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]                                                                                                                                       
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))  

###################################################################
def load_dataset(file_name, path, netcdf=False):
    """
    Load dataset; return truth and prediction arrays
    """
    # LOAD NETCDF DATASET
    datadir = os.path.join(path, "atmos_1day_d12240_plevel.nc")
    netcdf_data = netcdf_dataset(datadir)
    lat = netcdf_data.variables["lat"][:]
    lon = netcdf_data.variables["lon"][:]
    # pressures = netcdf_data.variables['level'][:]
    # print(pressures)
    print("lat: ", lat)
    print("lon: ", lon)

    # LOAD PICKLE DATASET
    dataset = from_pickle(os.path.join(path, file_name))
    targets = dataset["targets"].T.reshape(33,1440,64,128).swapaxes(1,0)
    predictions = dataset["predictions"].T.reshape(33,1440,64,128).swapaxes(1,0)

    if netcdf:
        targets = netcdf_data.variables["gwfu_cgwd"]
        print(type(targets))
    
    print("Predictions: ", predictions.shape)
    print("Targets : ", targets.shape)
    print(type(predictions))
    
    return targets, predictions, lon, lat
####################################################################

def three_panel_overlay(
    targets: np.array, 
    predictions: np.array,
    lon: np.array,
    lat: np.array,
    ) -> None:
    """
    Generates a three panel figure. The first panel is truth gwfu values, 
    the second panel is the difference between truth and predictions, and 
    the final panel is predictions. 

    Arguments: 
    Returns: None
    """
    projection_ccrs = ccrs.PlateCarree()
    #fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, subplot_kw={'projection': projection_ccrs})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, subplot_kw={'projection': projection_ccrs})

    dif = targets - predictions
    #dif = targets - predictions[:, :22,:,:]

    # Generate Slices
    dif_slice = dif[T, PLEVEL,:,:]
    targets_slice = targets[T, PLEVEL,:,:]
    pred_slice = predictions[T, PLEVEL,:,:]

    lon, lat = np.meshgrid(lon, lat)
    #pred_slice = np.ma.core.MaskedArray(pred_slice, mask=False, fill_value=1e+20)

    # import pdb
    # pdb.set_trace()

    # Create Colorbar
    vmax = np.max([targets_slice, pred_slice, dif_slice])
    vmin = np.min([targets_slice, pred_slice, dif_slice])
    vmax = np.max([np.abs(vmin), vmax])
    vmin = -vmax
    print("vmin ", vmin)
    print("vmax ", vmax)
    cmap = cm.get_cmap("BrBG", 64)
    norm = MidpointNormalize(midpoint=0.0, vmin=vmin, vmax=vmax)

    # Truth
    ax1.set_title("MiMA")
    ax1.set_extent((-180,180,-90,90), crs=projection_ccrs)
    ax1.coastlines()
    img1 = ax1.pcolormesh(lon, lat, targets_slice, transform=projection_ccrs, cmap=cmap, vmin=vmin, vmax=vmax)

    # Dif
    ax2.set_title("Difference")
    ax2.set_extent((-180,180,-90,90), crs=projection_ccrs)
    ax2.coastlines()
    img2 = ax2.pcolormesh(lon, lat, dif_slice, transform=projection_ccrs, cmap=cmap, vmin=vmin, vmax=vmax) 

    # Predictions
    ax3.set_title("ANN")
    ax3.set_extent((-180,180,-90,90), crs=projection_ccrs)
    ax3.coastlines()
    img3 = ax3.pcolormesh(lon, lat, pred_slice, transform=projection_ccrs, cmap=cmap, vmin=vmin, vmax=vmax) 

    # Plot Colorbar

    # Horizontal
    # fig.subplots_adjust(bottom=.2)
    # cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.025])
    # plt.colorbar(img1, cax=cbar_ax, orientation="horizontal")

    # Vertical
    # fig.subplots_adjust(left=.20)
    cbar_ax = fig.add_axes([0.15, 0.15, 0.025, .7])
    plt.colorbar(img1, cax=cbar_ax, orientation="vertical")
    # Labels    
    fig.suptitle("Zonal Gravity Wave Drag at ~100hPa", fontsize=22)
    plt.show()

def main():
    fp = "/home/zespinosa/Stanford/research/Coupling_MIMA_GW/netcdf_data"
    fn = "predictions.pkl"
    targets, predictions, lon, lat = load_dataset(fn, fp, netcdf=False)
    three_panel_overlay(
        targets=targets, 
        predictions=predictions, 
        lon=lon, 
        lat=lat
    )

if __name__ == '__main__':
    main()
