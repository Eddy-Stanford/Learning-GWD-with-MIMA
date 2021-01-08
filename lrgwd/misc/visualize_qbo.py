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

LAST_PLEVEL = 33 #26 18
LOWEST_PLEVEL = 2
FEAT = "ucomp"
FEAT = "gwfu_cgwd"

def true_qbo(
    filepath: Union[os.PathLike, str] = "/data/cees/zespinos/netcdf_data/MiMA-topo-v1.1-40-level",
) -> None:
    with netcdf.netcdf_file(
        os.path.join(filepath, "atmos_1day_d11160_plevel.nc")) as year_one_qbo, netcdf.netcdf_file(
        os.path.join(filepath, "atmos_1day_d11520_plevel.nc")) as year_two_qbo, netcdf.netcdf_file(
        os.path.join(filepath, "atmos_1day_d11880_plevel.nc")) as year_three_qbo, netcdf.netcdf_file(
        os.path.join(filepath, "atmos_1day_d12240_plevel.nc")) as year_four_qbo, netcdf.netcdf_file(
        os.path.join(filepath, "atmos_1day_d12600_plevel.nc")) as year_five_qbo:

        feat_data = [
            year_one_qbo.variables[FEAT][:,LOWEST_PLEVEL:LAST_PLEVEL,:,:],
            year_two_qbo.variables[FEAT][:,LOWEST_PLEVEL:LAST_PLEVEL,:,:],
            year_three_qbo.variables[FEAT][:,LOWEST_PLEVEL:LAST_PLEVEL, :,:],
            year_four_qbo.variables[FEAT][:,LOWEST_PLEVEL:LAST_PLEVEL,:,:],
            year_five_qbo.variables[FEAT][:,LOWEST_PLEVEL:LAST_PLEVEL,:,:]
        ]

        plevels = year_one_qbo.variables["level"][LOWEST_PLEVEL:LAST_PLEVEL]

        # Two weeks - 60 6 hour increments = 15 days
        two_week_windows = [60*i for i in range(24*len(feat_data)+1)]
        # xticks: every 4 months or 8 two week increments
        xticks= list(range(0, 24*len(feat_data), 8))
        # xtick_labels: every 4 months x add label
        xticks_labels = list(range(0, 12*len(feat_data), 4))
        feat_data = np.concatenate(feat_data, axis=0)

        feat_monthly_avgs = generate_monthly_averages(feat_data, two_week_windows)

        return feat_monthly_avgs, plevels, xticks, xticks_labels

def generate_monthly_averages(data, two_weeks):
        latitude = 32 # 32 = (equator) or 53 (polar vortex 60N) or 11 (south 60S)
        data_avgs = []
        for i in range(len(two_weeks)-1):
            lon_avg = []
            for j in range(128):
                vertical_column_avg = np.average(data[two_weeks[i]:two_weeks[i+1]-1, :, latitude, j], axis=0)
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
    fig, axes = plt.subplots(nrows=3)

    vmax = 1e-4
    print("max: ", np.max([np.max(truth), np.max(predictions)]))
    vmin = -1e-4
    print("min: ", np.min([np.min(truth), np.min(predictions)]))

    axes_flat = axes.flat
    truth_ax = axes_flat[0]
    diff_ax = axes_flat[1]
    pred_ax = axes_flat[2]

    colormap = 'RdYlBu'
    #cmap = cm.get_cmap("BrBG", 128)
    #cmap = cm.get_cmap("terrain")
    cmap = cm.get_cmap(colormap)
    #cmaprange = range(0, cmap.N, 16)
    #cmaplist = [cmap(i) for i in cmaprange]
    #cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(list(cmaprange)))

    linthresh = 1e-7 #5e-8
    linscale = 1.0
    base = 10

    print("plevels: ", plevels)
    Y = plevels
    X = np.arange(120)
    X,Y = np.meshgrid(X,Y)

    img1 = truth_ax.pcolor(X,Y,truth, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    img2 = pred_ax.pcolor(X,Y,predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    img3 = diff_ax.pcolor(X,Y,truth-predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))

    #img1 = truth_ax.imshow(truth, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    #img2 = pred_ax.imshow(predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    #img3 = diff_ax.imshow(truth-predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))

    truth_ax.set_yscale('log')
    diff_ax.set_yscale('log')
    pred_ax.set_yscale('log')

    diff_ax.set_ylim(np.max(plevels), np.min(plevels))
    pred_ax.set_ylim(np.max(plevels), np.min(plevels))
    truth_ax.set_ylim(np.max(plevels), np.min(plevels))

    truth_ax.set_yticks([1, 10, 100])
    diff_ax.set_yticks([1, 10, 100])
    pred_ax.set_yticks([1, 10, 100])

    truth_ax.set_yticklabels([1, 10, 100])
    diff_ax.set_yticklabels([1, 10, 100])
    pred_ax.set_yticklabels([1, 10, 100])

    """
    img1 = truth_ax.imshow(truth, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    img2 = pred_ax.imshow(predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    img3 = diff_ax.imshow(truth-predictions, vmin=vmin, vmax=vmax, cmap=cmap, norm=colors.SymLogNorm(linthresh=linthresh, linscale=linscale, base=base, vmin=vmin, vmax=vmax))
    """
    #norm=MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax))

    labelsize=14
    axlabelsize=16

    truth_ax.set_ylabel("Pressure [hPa]", fontsize=axlabelsize)
    pred_ax.set_ylabel("Pressure [hPa]", fontsize=axlabelsize)
    diff_ax.set_ylabel("Pressure [hPa]", fontsize=axlabelsize)

    # Set y axes ticks
    #truth_ax.set_yticks(ticks=[0,12,24]) #list(range(0, len(plevels), 4)))
    #pred_ax.set_yticks(ticks=[0,12,24]) #list(range(0, len(plevels), 4)))
    #diff_ax.set_yticks(ticks=[0,12,24]) #list(range(0, len(plevels), 4)))
    # Set y axes labels
    #truth_ax.set_yticklabels([1.0, 10.0, 100.0]) #plevels[::4])
    #pred_ax.set_yticklabels([1.0, 10.0, 100.0]) #plevels[::4])
    #diff_ax.set_yticklabels([1.0, 10.0, 100.0]) #plevels[::4])

    # Set vertical dashed lines - delineate year
    truth_ax.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    truth_ax.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    truth_ax.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    truth_ax.axvline(x=96, color='black', alpha=.5, linestyle="dashed")

    pred_ax.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    pred_ax.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    pred_ax.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    pred_ax.axvline(x=96, color='black', alpha=.5, linestyle="dashed")

    diff_ax.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    diff_ax.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    diff_ax.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    diff_ax.axvline(x=96, color='black', alpha=.5, linestyle="dashed")


    # Set only x ticks on truth and diff
    diff_ax.set_xticks(xticks)
    truth_ax.set_xticks(xticks)
    diff_ax.get_xaxis().set_ticklabels([])
    truth_ax.get_xaxis().set_ticklabels([])
    #truth_ax.set_xlabel("Months", fontsize=axlabelsize)
    #truth_ax.set_xticklabels(xticks_labels)

    # Set x ticks and labels on pred
    pred_ax.set_xticks(xticks)
    pred_ax.set_xlabel("Months", fontsize=axlabelsize)
    pred_ax.set_xticklabels(xticks_labels)

    # Set colobar
    cbar = fig.colorbar(img2, ax=axes.ravel().tolist())
    cbar.extend = 'both'
    #ticks = np.insert(np.linspace(-7e-5, 7e-5, 8), [4], 0)
    #cbar.set_ticks(ticks)
    cbar.set_label(r"[m$s^{-2}$]", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Set title
    truth_ax.set_title("Zonal Equatorial Gravity Wave Drag Tendencies", fontsize="x-large")

    truth_ax.tick_params(axis='y', labelsize=labelsize)
    diff_ax.tick_params(axis='y', labelsize=labelsize)
    pred_ax.tick_params(axis='both', labelsize=labelsize)

    # Add Labels text
    textsize = 16
    truth_ax.text(.01, .9, "a) AD99", transform=truth_ax.transAxes, fontsize=textsize)
    diff_ax.text(.01, .9, "b) Difference", transform=diff_ax.transAxes, fontsize=textsize)
    pred_ax.text(.01, .9, "c) ANN", transform=pred_ax.transAxes, fontsize=textsize)


    fig.set_size_inches(16,9)
    plt.savefig(f"qbo_five_years_{colormap}_exp.png")
    plt.savefig(f"qbo_five_years_{colormap}_exp.pdf")

def predicted_qbo(
    plevels: List[float],
    filepath: Union[os.PathLike, str] = "/data/cees/zespinos/runs/feature_experiments/40_levels",
):

    years = ["year_two", "year_one", "year_three", "year_four", "year_five"]
    year_predictions = []
    for year in years:
        year_metrics = from_pickle(os.path.join(filepath, f"{year}/evaluate/gwfu/full_features/metrics.pkl"))
        year_data = from_pickle(os.path.join(filepath, f"{year}/evaluate/gwfu/full_features/predictions.pkl"))
        year_data = year_data["predictions"].T
        year_data = year_data.reshape(33, 1440, 64, 128).swapaxes(1, 0)
        year_predictions.append(year_data[:,LOWEST_PLEVEL:LAST_PLEVEL, :,:])
        print(f"{year}: ", year_metrics["r_squared"])

    year_data = None
    months = [60*i for i in range(24*len(year_predictions)+1)]

    year_predictions = np.concatenate(year_predictions, axis=0)

    year_predictions = generate_monthly_averages(year_predictions, months)

    return year_predictions


def main():
        # Plot True QBO
        print("############## starting script ##############")
        year_targets, plevels, xticks, xlabels = true_qbo()
        print("############## load true qbo ##############")
        #plot_qbo(year_targets.T, plevels, xticks, xlabels)

        # Generate Predicted QBO
        # year_targets = targets_qbo(plevels)
        year_predictions = predicted_qbo(plevels)
        print("############## get predicted qbo  ##############")
        plot_truth_vs_predictions(year_targets.T, year_predictions.T, plevels, xticks, xlabels)
        print("############## plot truth and predictions ##############")


        # Plot Predicted QBO
        # plot_qbo(year_one, plevels, xticks)

main()
