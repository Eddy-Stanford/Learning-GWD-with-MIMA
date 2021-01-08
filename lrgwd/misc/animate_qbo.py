import os
from typing import Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.io import netcdf

from lrgwd.utils.io import from_pickle

LAST_PLEVEL = 28 #18
LOWEST_PLEVEL = 12
FEAT = "gwfu_cgwd"

DURATION=15
T=0
LAT = 32  # Equator

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


def create_linear_segmented_colorbar(n=20, cmap="BrBG"):
    plt.cm["BrBG"]
    cmaplist = []


def get_plevels(
    filepath: Union[os.PathLike, str] = "/data/cees/zespinos/netcdf_data/MiMA-topo-v1.1-40-level",
) -> None:
    """
    Return the pressure levels from netcdf data
    """
    with netcdf.netcdf_file(os.path.join(filepath, "atmos_1day_d12240_plevel.nc")) as year_four_qbo:
        plevels = year_four_qbo.variables["level"][:]#[LOWEST_PLEVEL:LAST_PLEVEL]
    return plevels


def get_eval(path):
    data = from_pickle(path)
    pred = data["predictions"].T
    pred = pred.reshape(33, 1440, 64, 128).swapaxes(1, 0)

    targets = data["targets"].T
    targets = targets.reshape(33, 1440, 64, 128).swapaxes(1, 0)

    return targets, pred


def get_evaluation_package(
    filepath: Union[os.PathLike, str] = "/data/cees/zespinos/runs/feature_experiments/40_levels",
):
    """
    Return targets and preditions from given evaluation path
    """

    targets_one, pred_one = get_eval(os.path.join(filepath, f"year_four/evaluate/gwfu/full_features/predictions.pkl"))
    targets_two, pred_two = get_eval(os.path.join(filepath, f"year_five/evaluate/gwfu/full_features/predictions.pkl"))

    return np.concatenate([targets_one, targets_two]), np.concatenate([pred_one, pred_two])


def get_tendency_slice(
        #targets: np.ndarray,
        #predictions: np.ndarray
):
    """
    Returns a vertical-equatorial tendency profile from targets and preditions of duration=DURATION
    """

    global T
    global targets
    global predictions

    target_slice = np.squeeze(targets[T:T+DURATION, LOWEST_PLEVEL:LAST_PLEVEL, LAT, :])
    pred_slice  = np.squeeze(predictions[T:T+DURATION, LOWEST_PLEVEL:LAST_PLEVEL, LAT, :])

    target_slice = np.mean(target_slice, axis=0)
    pred_slice = np.mean(pred_slice, axis=0)

    T+=DURATION

    return target_slice, pred_slice


def  setup_tendency_fig(targets, predictions):
    fig = plt.figure(figsize=(16,9))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2,1),
                    axes_pad=0.25,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="2%",
                    cbar_pad=.15,
                    )
    ax1, ax2 = axs
    target_slice, pred_slice = get_tendency_slice()

    #vmin = np.min([predictions, targets])
    #vmax = np.max([predictions, targets])
    #vmax = np.max([np.abs(vmin), vmax])
    #vmin = -vmax

    vmin = np.min([pred_slice, target_slice])
    vmax = np.max([pred_slice, target_slice])
    vmax = np.max([np.abs(vmin), vmax])
    vmin = -vmax

    cmap = cm.get_cmap("BrBG", 32)
    #midnorm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)
    cnorm = colors.SymLogNorm(linthresh=10e-7, vmin=vmin, vmax=vmax)

    axlabelsize=12

    # Targets
    img1 = ax1.imshow(target_slice, vmin=vmin, vmax=vmax, cmap=cmap, norm=cnorm)
    ax1.set_ylabel("Pressure (hPa)", fontsize=axlabelsize)
    # Set Y Labels
    ax1.set_yticks(ticks=[0, 3.5, 7, 10.5, 14])
    ax1.set_yticklabels([10, 40, 70, 100, 130])

    # Predictions
    img2 = ax2.imshow(pred_slice, vmin=vmin, vmax=vmax, cmap=cmap, norm=cnorm)
    ax2.set_ylabel("Pressure (hPa)", fontsize=axlabelsize)

    # Set Y Labels Fix Bad Labels
    ax2.set_yticks(ticks=[0, 3.5, 7, 10.5, 14])
    ax2.set_yticklabels([10, 40, 70, 100, 130])

    # Set X Labels
    ax2.set_xlabel("Longitude", fontsize=axlabelsize)
    xticklabels = np.arange(0,180,20)
    ax2.set_xticklabels(xticklabels)
    ax2.set_xticks(ticks=[xtick*(120/180) for xtick in xticklabels])

    # Colorbar
    #cbar = axs.cbar_axes[0].colorbar(img2, extend='neither')
    cbar = fig.colorbar(img2, cax=axs.cbar_axes[0])
    #cbar.ax.set_ylabel(r'm/ $s^2$')
    #ticks = np.insert(np.linspace(-7e-5, 7e-5, 8), [4], 0)
    #cbar.set_ticks(ticks)


    ax1.set_title("MiMA")
    ax2.set_title("ANN")


#    plt.savefig("equator_tendency_slice.png")
    return img1, img2, fig


def update_tendency(self):
    target_slice, pred_slice = get_tendency_slice()
    img1.set_data(target_slice)
    img2.set_data(pred_slice)

    return img1, img2

plevels = get_plevels()
print(plevels)
targets, predictions = get_evaluation_package()
img1, img2, fig = setup_tendency_fig(targets, predictions)
simulation = animation.FuncAnimation(fig, update_tendency, blit=False, frames=int(np.floor(2*1440/DURATION)), interval=50)
simulation.save('qbo_tendency_time_lapse.mp4')
