from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

from lrgwd.utils.io import from_pickle
import pandas as pd

def true_qbo():
    with netcdf.netcdf_file("../../../netcdf_data/atmos_1day_d11160_plevel.nc") as year_one_qbo, netcdf.netcdf_file(
        "../../../netcdf_data/atmos_1day_d11520_plevel.nc") as year_two_qbo ,netcdf.netcdf_file(
        "../../../netcdf_data/atmos_1day_d12240_plevel.nc") as year_three_qbo ,netcdf.netcdf_file(
        "../../../netcdf_data/atmos_1day_d11880_plevel.nc") as year_four_qbo:

        ucomp_data = [year_one_qbo.variables["gwfu_cgwd"][:,:18,:,:], year_two_qbo.variables["gwfu_cgwd"][:,:18,:,:] ,year_three_qbo.variables["gwfu_cgwd"][:, :18, :,:], year_four_qbo.variables["gwfu_cgwd"][:,:18,:,:]]

        plevels = year_one_qbo.variables["level"][:]
        months = [60*i for i in range(24*len(ucomp_data)+1)]
        xticks= list(range(0, 24*len(ucomp_data), 2))
        xticks_labels = list(range(0, 12*len(ucomp_data)))
        ucomp_data = np.concatenate(ucomp_data, axis=0)

        ucomp_monthly_avgs = generate_monthly_averages(ucomp_data, months)

        return ucomp_monthly_avgs, plevels, xticks, xticks_labels

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
    img = plt.imshow(data, cmap="BrBG")
    cbar = plt.colorbar(img, shrink=.5)
    cbar.set_label("ucomp (m/s)")
    plt.xlabel("Months")
    plt.xticks(xticks, labels=xticks_labels)
    plt.ylabel("Pressure (hPa)")
    plt.yticks(ticks=list(range(len(plevels))), labels=plevels)


    plt.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    plt.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    plt.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    # plt.xlim(left=1, right=xticks[len(xticks)-1])
    plt.title("Mima UComp Trends [4 years]")
    plt.show()

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


def plot_truth_vs_predictions(truth, predictions, plevels, xticks, xticks_labels):
    fig, axes = plt.subplots(nrows=2, sharex=True)

    vmax = 7e-5 #np.max([np.max(truth), np.max(predictions)]) 
    vmin = -7e-5 #np.min([np.min(truth), np.min(predictions)]) 

    axes_flat = axes.flat
    truth_ax = axes_flat[0] 
    pred_ax = axes_flat[1]

    img1 = truth_ax.imshow(truth, vmin=vmin, vmax=vmax, cmap="BrBG", norm=MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax))
    img2 = pred_ax.imshow(predictions, vmin=vmin, vmax=vmax, cmap="BrBG", norm=MidpointNormalize(midpoint=0,vmin=vmin, vmax=vmax))

    truth_ax.set_ylabel("Pressure (hPa)")
    pred_ax.set_ylabel("Pressure (hPa)")
    truth_ax.set_yticks(ticks=list(range(len(plevels))))
    pred_ax.set_yticks(ticks=list(range(len(plevels))))
    truth_ax.set_yticklabels(plevels)
    pred_ax.set_yticklabels(plevels)

    truth_ax.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    truth_ax.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    truth_ax.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    pred_ax.axvline(x=24, color='black', alpha=.5, linestyle="dashed")
    pred_ax.axvline(x=48, color='black', alpha=.5, linestyle="dashed")
    pred_ax.axvline(x=72, color='black', alpha=.5, linestyle="dashed")
    cbar = fig.colorbar(img2, ax=axes.ravel().tolist())
    plt.xlabel("Months")
    plt.xticks(ticks=xticks, labels=xticks_labels)
    cbar.set_label("gwfu (m/s^2)")
    truth_ax.set_title("Truth GWFU Trends")
    pred_ax.set_title("Predictions GWFU Trends")
    plt.show()


def targets_qbo(plevels):
    # year_one_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_one/metrics.pkl")
    year_one_targets = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_one/predictions.pkl")
    year_one_targets = year_one_targets["targets"].T

    # year_two_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_two/metrics.pkl")
    year_two_targets = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_two/predictions.pkl")
    year_two_targets = year_two_targets["targets"].T

    # year_three_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_three/metrics.pkl")
    year_three_targets = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_three/predictions.pkl")
    year_three_targets = year_three_targets["targets"].T


    year_one_targets = year_one_targets.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    year_two_targets = year_two_targets.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    year_three_targets = year_three_targets.reshape(18, 1440, 64, 128).swapaxes(1, 0)

    year_targets = [year_one_targets, year_two_targets, year_three_targets]

    months = [60*i for i in range(24*len(year_targets)+1)]
    xticks= list(range(12*len(year_targets)))

    year_targets = np.concatenate(year_targets, axis=0)
    year_one_targets, year_two_targets, year_three_targets = None, None, None


    year_targets = generate_monthly_averages(year_targets, months)
    # plot_qbo(year_predictions, months, xticks)
    # plot_qbo(year_targets, months, xticks)
    
    # year_total = np.concatenate([year_targets.T, year_predictions.T], axis=0)
    # months = [60*i for i in range(24*4+1)]
    # xticks= list(range(0, 36*2, 2))
    # xticks_labels = list(range(0, 36))

    return year_targets
    # plot_truth_vs_predictions(year_targets.T, year_predictions.T, plevels, xticks, xticks_labels)

    # return year_total, months, xticks

def predicted_qbo(plevels):
    # year_one_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_one/metrics.pkl")
    year_one_data = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_one/predictions.pkl")
    year_one_predictions = year_one_data["predictions"].T
    # year_one_targets = year_one_data["targets"].T
    year_one_data = None

    # year_two_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_two/metrics.pkl")
    year_two_data = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_two/predictions.pkl")
    year_two_predictions = year_two_data["predictions"].T
    # year_two_targets = year_two_data["targets"].T
    year_two_data = None

    # year_three_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_three/metrics.pkl")
    year_three_data = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_three/predictions.pkl")
    year_three_predictions = year_three_data["predictions"].T
    # year_three_targets = year_three_data["targets"].T
    year_three_data = None

    year_four_metrics = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_four/metrics.pkl")
    year_four_data = from_pickle("../../runs/Logcosh_DeepNet/evaluate_year_four/predictions.pkl")
    year_four_predictions = year_four_data["predictions"].T
    # year_three_targets = year_three_data["targets"].T
    year_four_data = None

    # print("Year_one: ", year_one_metrics["r_squared"])
    # print("Year_two: ", year_two_metrics["r_squared"])
    # print("Year_three: ", year_three_metrics["r_squared"])
    print("Year_four: ", year_four_metrics["r_squared"])

    year_one_predictions = year_one_predictions.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    year_two_predictions = year_two_predictions.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    year_three_predictions = year_three_predictions.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    year_four_predictions = year_four_predictions.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    # year_one_targets = year_one_targets.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    # year_two_targets = year_two_targets.reshape(18, 1440, 64, 128).swapaxes(1, 0)
    # year_three_targets = year_three_targets.reshape(18, 1440, 64, 128).swapaxes(1, 0)

    year_predictions = [year_one_predictions, year_two_predictions, year_three_predictions, year_four_predictions] 
    # year_targets = [year_one_targets, year_two_targets, year_three_targets]

    months = [60*i for i in range(24*len(year_predictions)+1)]
    xticks_labels = list(range(12*len(year_predictions)))
    xticks= list(range(0, 12*len(year_predictions)*2, 2))

    year_predictions = np.concatenate(year_predictions, axis=0)
    year_one_predictions, year_two_predictions, year_three_predictions, year_four_predictions = None, None, None, None
    # year_targets = np.concatenate(year_targets, axis=0)


    year_predictions = generate_monthly_averages(year_predictions, months)
    # year_targets = generate_monthly_averages(year_targets, months)
    # plot_qbo(year_predictions, months, xticks)
    # plot_qbo(year_targets, months, xticks)
    
    # year_total = np.concatenate([year_targets.T, year_predictions.T], axis=0)
    # months = [60*i for i in range(24*4+1)]

    return year_predictions, xticks, xticks_labels
    # plot_truth_vs_predictions(year_targets.T, year_predictions.T, plevels, xticks, xticks_labels)

    # return year_total, months, xticks


def main():
        # Plot True QBO
        year_targets, plevels, xticks, xlabels = true_qbo()
        # plot_qbo(ucomp_data, plevels, xticks, xlabels)

        # Generate Predicted QBO 
        # year_targets = targets_qbo(plevels)
        year_predictions, xticks, xticks_labels = predicted_qbo(plevels)
        plot_truth_vs_predictions(year_targets.T, year_predictions.T, plevels, xticks, xticks_labels)


        # Plot Predicted QBO
        # plot_qbo(year_one, plevels, xticks)

main()


    # year_one_predictions = pd.read_csv("../../runs/data/year_one/extracted_unshuffled/gwfu.csv", header=None).to_numpy().T
    # zeros = np.zeros((1,18))
    # year_one_predictions = np.concatenate([year_one_predictions, zeros], axis=0)