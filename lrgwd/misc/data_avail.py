import os
from typing import Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.io import netcdf

from lrgwd.utils.io import from_pickle

LAST_PLEVEL = 26 #18
LOWEST_PLEVEL = 0
FEAT = "gwfu_cgwd"

def get_plevels(
    filepath: Union[os.PathLike, str] = "/data/cees/zespinos/netcdf_data/MiMA-topo-v1.1-40-level",
):
    with netcdf.netcdf_file(os.path.join(filepath, "atmos_1day_d11160_plevel.nc")) as year_one_qbo:
        plevels = year_one_qbo.variables["level"][LOWEST_PLEVEL:LAST_PLEVEL]
    return plevels

def get_ticks(num_years=3):
    xticks= list(range(0, 24*num_years, 8))
    xticks_labels = list(range(0, 12*num_years, 4))
    return xticks, xticks_labels

def generate_monthly_averages(data, months):
        data_avgs = []
        for i in range(len(months)-1):
            lon_avg = []
            for j in range(128):
                vertical_column_avg = np.average(data[months[i]:months[i+1]-1, :, 32, j], axis=0)
                lon_avg.append(vertical_column_avg)
            data_avgs.append(np.average(lon_avg, axis=0))

        data_avgs = np.array(data_avgs)

        return data_avgs

def plot_qbo(data, plevels, xticks, xticks_labels):
    fig = plt.figure()
    vmin = -50
    vmax = 50
    img = plt.imshow(data, vmin=vmin, vmax=vmax, cmap="BrBG", norm=MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(img, shrink=.5)
    cbar.set_label("ucomp (m/s)")
    plt.xlabel("Months")
    plt.xticks(xticks, labels=xticks_labels)
    plt.ylabel("Pressure (hPa)")
    plt.yticks(ticks=list(range(0, len(plevels), 2)), labels=plevels[::2])


    plt.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    plt.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    plt.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    plt.axvline(x=96, color='black', alpha=.5, linestyle="dashed")
    # plt.xlim(left=1, right=xticks[len(xticks)-1])
    plt.title("QBO: 15 Day Mean Zonal Wind (MiMA)")
    fig.set_size_inches(32,18)

    plt.savefig("ucomp_qbo_five_years.png")

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

def plot_truth_vs_predictions(truth, predictions, plevels, xticks, xticks_labels):
    fig, axes = plt.subplots(ncols=2)

    vmax = 7e-5 #np.max([np.max(truth), np.max(predictions)])
    vmin = -7e-5 #np.min([np.min(truth), np.min(predictions)])

    axes_flat = axes.flat
    truth_ax = axes_flat[0]
    pred_ax = axes_flat[1]

    cmap = cm.get_cmap("BrBG", 32)
    #cmaprange = range(0, cmap.N, 16)
    #cmaplist = [cmap(i) for i in cmaprange]
    #cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(list(cmaprange)))

    img1 = truth_ax.imshow(truth, vmin=vmin, vmax=vmax, cmap=cmap, norm=MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax))
    img2 = pred_ax.imshow(predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax))

    labelsize=12
    axlabelsize=14

    truth_ax.set_ylabel("Pressure (hPa)", fontsize=axlabelsize)
    pred_ax.set_ylabel("Pressure (hPa)", fontsize=axlabelsize)
    truth_ax.set_yticks(ticks=[0,12,24]) #list(range(0, len(plevels), 4)))
    pred_ax.set_yticks(ticks=[0,12,24]) #list(range(0, len(plevels), 4)))
    truth_ax.set_yticklabels([1.0, 10.0, 100.0]) #plevels[::4])
    pred_ax.set_yticklabels([1.0, 10.0, 100.0]) #plevels[::4])

    truth_ax.axhline(y=24, color='black', alpha=.5, linestyle="solid")
    truth_ax.axhline(y=48, color='black', alpha=.5, linestyle="solid")
    truth_ax.axhline(y=72, color='black', alpha=.5, linestyle="solid")
    truth_ax.axhline(y=96, color='black', alpha=.5, linestyle="solid")
    pred_ax.axhline(y=24, color='black', alpha=.5, linestyle="solid")
    pred_ax.axhline(y=48, color='black', alpha=.5, linestyle="solid")
    pred_ax.axhline(y=72, color='black', alpha=.5, linestyle="solid")
    pred_ax.axhline(y=96, color='black', alpha=.5, linestyle="solid")

    cbar = fig.colorbar(img2, ax=axes.ravel().tolist())
    ticks = np.insert(np.linspace(-7e-5, 7e-5, 8), [4], 0)
    cbar.set_ticks(ticks)

    pred_ax.set_xlabel("Months", fontsize=axlabelsize)
    pred_ax.set_xticks(xticks)
    pred_ax.set_xticklabels(xticks_labels)

    truth_ax.set_xlabel("Months", fontsize=axlabelsize)
    truth_ax.set_xticks(xticks)
    truth_ax.set_xticklabels(xticks_labels)

    cbar.set_label("gwfu (m/s^2)", fontsize=axlabelsize)
    truth_ax.set_title("Physics-Based: Zonal Gravity Wave Tendencies", fontsize="x-large")
    pred_ax.set_title("Data-Driven: Zonal Gravity Wave Tendencies", fontsize="x-large")


    truth_ax.tick_params(axis='both', labelsize=labelsize)
    pred_ax.tick_params(axis='both', labelsize=labelsize)


    fig.set_size_inches(16,9)
    plt.savefig("gwfd_qbo_five_years.png")

def get_predictions(
    plevels: List[float],
    filepath: Union[os.PathLike, str] = "/data/cees/zespinos/runs/feature_experiments/40_levels/year_three/evaluate/gwfu/data_avail",
):
    months = [60*i for i in range(24*1+1)]
    data_avail = ["three", "six", "nine", "full_features"]
    data_predictions  = []
    for avail in data_avail:
        avail_metrics = from_pickle(os.path.join(filepath, avail, "metrics.pkl"))
        avail_pred= from_pickle(os.path.join(filepath, avail, "predictions.pkl"))
        avail_truth = avail_pred["targets"].T
        avail_truth = avail_truth.reshape(33, 1440, 64, 128).swapaxes(1, 0)
        avail_truth = avail_truth[:,LOWEST_PLEVEL:LAST_PLEVEL, :,:]

        avail_pred = avail_pred["predictions"].T
        avail_pred = avail_pred.reshape(33, 1440, 64, 128).swapaxes(1, 0)
        avail_pred = avail_pred[:,LOWEST_PLEVEL:LAST_PLEVEL, :,:]

        avail_pred = generate_monthly_averages(avail_pred, months)
        data_predictions.append(avail_pred)
        print(f"{avail}: ", avail_metrics["r_squared"])

    avail_pred = None
    avail_truth = generate_monthly_averages(avail_truth, months)
    data_predictions.append(avail_truth)
    data_predictions = np.concatenate(data_predictions, axis=1)

    return data_predictions

def main():
        plevels = get_plevels()
        xticks, xlabels = get_ticks(num_years=3)

        print("I got ticks")
        predictions = get_predictions(plevels)
        print("I got predictions")
        plot_truth_vs_predictions(predictions.T, predictions.T, plevels, xticks, xlabels)

main()


