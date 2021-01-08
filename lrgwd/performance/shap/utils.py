import os
from typing import Dict, List, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error

from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle, to_pickle
from lrgwd.utils.logger import logger


class EvaluationPackage(object):
    def __init__(self,
        source_path: Union[os.PathLike, str],
        scaler_path: Union[os.PathLike, str],
        num_samples: Union[None, int],
        target: str,
        save_path: Union[os.PathLike, str],
        model,
    ) -> None:

        X_fp = os.path.join(source_path, "tensors.csv")
        Y_fp  = os.path.join(source_path,  f"{target}.csv")

        # Get Scalers
        X_scaler_fp = os.path.join(scaler_path, "tensors_scaler.pkl")
        self.X_scaler = from_pickle(X_scaler_fp)

        Y_scaler_fp = os.path.join(scaler_path, f"{target}_scaler.pkl")
        self.Y_scaler = from_pickle(Y_scaler_fp)

        for X, Y in zip(
                pd.read_csv(X_fp, header=None, chunksize=num_samples),
                pd.read_csv(Y_fp, header=None, chunksize=num_samples)
        ):

            # Tensors
            self.X_raw = X.to_numpy()
            self.X = self.X_scaler.transform(self.X_raw)

            # Targets
            self.Y_raw = Y.to_numpy()
            self.Y = self.Y_scaler.transform(self.Y_raw)

            # Predictions
            self.Y_pred = self.predict(model)

            return

    def predict(self, model):
        predictions = model.predict(self.X)
        predictions = np.hstack(predictions)
        predictions = self.Y_scaler.inverse_transform(predictions)

        return predictions


    def split_predictions_on_plevel(self,
        predictions: np.ndarray,
        targets: np.ndarray,
        outliers: Union[None, float]
    ):
        # Split predictions per level
        plevel_predictions = {}
        plevel_targets = {}

        num_plevels = predictions.shape[1]
        for i in range(num_plevels):
            slice_predictions = predictions[:, i]
            slice_targets = targets[:, i]

            # Remove Outliers
            if outliers is not None:
                plevel_predictions, plevel_targets = self.remove_outliers(
                    predictions=slice_predictions,
                    targets=slice_targets,
                    outliers=float(outliers),
                )

            plevel_predictions[f"plevel_{i}"] = slice_predictions
            plevel_targets[f"plevel_{i}"] = slice_targets

        return plevel_predictions, plevel_targets


    def remove_outliers(self,
        predictions: np.ndarray,
        targets: np.ndarray,
        outliers: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_outliers = is_outlier(targets, thresh=outliers)
        predictions = predictions[~target_outliers]
        targets = targets[~target_outliers]

        prediction_outliers = is_outlier(predictions, thresh=outliers)
        predictions = predictions[~prediction_outliers]
        targets = targets[~prediction_outliers]

        return (predictions, targets)


def generate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    plevel_predictions: Dict[str, np.ndarray],
    plevel_targets: Dict[str, np.ndarray],
    save_path: Union[os.PathLike, str]
) -> None:
    # Pressure Level Specific Metrics
    metrics = {
        "maes": mean_absolute_error(targets, predictions, multioutput="raw_values"),
        "rmse": mean_squared_error(targets, predictions, multioutput="raw_values", squared=False),
        "stds": np.std(targets, axis=1),
        "mins": np.min(targets, axis=1),
        "maxes": np.max(targets, axis=1),
        "means": np.mean(targets, axis=1),
        "medians": np.median(targets, axis=1),
    }

    metrics["r_squared"] = calculate_r_squared(
        test_predictions=plevel_predictions,
        test_targets=plevel_targets,
    )

    to_pickle(os.path.join(save_path, "metrics.pkl"), metrics)

    return metrics


def calculate_r_squared(
    test_predictions: Dict[str, np.ndarray],
    test_targets: Dict[str, np.ndarray],
) -> List[float]:
    r_squared = []
    for i, values in enumerate(zip(test_predictions.values(), test_targets.values())):
        predictions, targets = values
        slope, intercept, r_value, p_value, std_err = linregress(predictions, targets)
        r_squared.append(r_value**2)
    return r_squared

def calculate_rsquared(
    Y_pred: np.ndarray,
    Y: np.ndarray,
    plevels: List[float],
) -> List[float]:
    r_squared = []
    for i in range(len(plevels)):
        y_pred = Y_pred[:, i]
        y = Y[:, i]
        slope, intercept, r_value, p_value, std_err = linregress(y_pred, y)
        r_squared.append(r_value**2)
    return r_squared

def plot_predictions_vs_truth(
    Y_pred: np.ndarray,
    Y: np.ndarray,
    plevels: List[float],
    r_squared: List[float],
    save_path: Union[os.PathLike, str] = "/lrgwd/performance/shap",
) -> None:
    fig, axs = plt.subplots(ncols=3, nrows=2)
    gs = axs[0,1].get_gridspec()
    r_squared.reverse()

    pidxs = [13, 20, 23, 27]
    plevels_labels = [10, 50, 100, 200]
    colors = cm.rainbow(np.linspace(0,1, len(plevels_labels)))

    # remove the underlying axes for axrsq
    for ax in axs[0:, 2]:
        ax.remove()
    axrsq = fig.add_subplot(gs[0:, 2])
    plt.subplots_adjust(wspace=.20, hspace=.30)

    labelsize=32
    panels = ["a", "b", "c", "d"]

    for pane, pkg in enumerate(zip(pidxs, plevels_labels)):
        pidx, plevel = pkg
        row, col = np.unravel_index(pane, (2,2))
        ax = axs[row, col]

        y = Y[:, pidx]
        y_pred = Y_pred[:, pidx]

        minlim = np.min([np.min(y), np.min(y_pred)])
        maxlim = np.max([np.max(y), np.max(y_pred)])
        maxlim = np.max([np.abs(minlim), maxlim])
        maxlim = maxlim + maxlim*.2 # Give some extra space
        minlim = -maxlim
        lims = [minlim, maxlim]

        ax.scatter(y, y_pred, color=colors[pane], label=f'{plevel} hPa')
        ax.set_xlim(minlim, maxlim)
        ax.set_ylim(minlim, maxlim)
        ax.ticklabel_format(style='sci')
        ax.grid(True, linewidth=2)
        ax.plot(lims, lims, linewidth=4)
        ax.set_title(f'{plevel} hPa', fontsize=36)
        ax.tick_params(labelsize=labelsize)

        yoffset = ax.yaxis.get_offset_text()
        yoffset.set_size(labelsize)

        xoffset = ax.xaxis.get_offset_text()
        xoffset.set_size(labelsize)

        cur_r_squared = np.round(r_squared[pidx], 3)
        ax.text(.85,.05,
            f"{panels[pane]})",
            transform=ax.transAxes,
            fontsize=48,
        )
        ax.text(
            0.1, 0.85,
            f"$R^2$: {cur_r_squared}",
            transform=ax.transAxes,
            fontsize=labelsize,
            bbox=dict(facecolor='none', edgecolor='black', pad=10.0)
        )

    # Add X and Y Axes Labels (Quad)
    ax1 = axs[1,0]
    size = 32
    fontdict = {
        'size': 36
    }

    ax1.text(
        -.25, -0.3, "AD99 [$ms^{-2}$]", fontdict=fontdict, transform=ax.transAxes
    )
    ax2 = axs[0,0]
    ax2.text(
        -1.5, .9, "ANN [$ms^{-2}$]", fontdict=fontdict, rotation=90, transform=ax.transAxes
    )

    # Add X and Y Axes Labels (Height)
    axrsq.set_ylabel("Pressure [hPa]", fontsize=36)
    axrsq.set_xlabel("$R^2$", fontsize=36)
    axrsq.tick_params(labelsize=size)
    axrsq.plot(r_squared, plevels, linewidth=4)
    axrsq.set_title("$R^2$ by height", fontsize=36)
    axrsq.set_yscale('log')
    axrsq.yaxis.set_label_position("right")
    axrsq.yaxis.tick_right()
    axrsq.text(
        .05, .03,
        "e)",
        transform=axrsq.transAxes,
        fontsize=48,
    )

    #axrsq.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    #axrsq.set_yticks([1.0, 5.0, 10.0, 50.0, 100.0, 500.0])

    pmax = max(plevels) + max(plevels)*.30
    pmin = min(plevels) - min(plevels)*.30
    axrsq.set_ylim(pmax, pmin)
    axrsq.set_yticks([.1,.5,1,5,10,50,100,200, 500])
    axrsq.set_yticklabels([".1", ".5", "1", "5", "10", "50", "100", "200", "500"])
    axrsq.set_xlim(.5, 1.0)
    axrsq.grid(True, linewidth=2)


    # Make figures full screen
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(os.path.join(save_path, "lrgwd/performance/shap", f"zonal_predictions_vs_truth_AD99.pdf"))
    fig.savefig(os.path.join(save_path, "lrgwd/performance/shap", f"zonal_predictions_vs_truth_AD99.png"))
    plt.close(fig)

"""
def plot_predictions_vs_truth(
    Y_pred: np.ndarray,
    Y: np.ndarray,
    plevels: List[float],
    save_path: Union[os.PathLike, str] = "/lrgwd/performance/shap",
) -> None:
    print("Plot predictions vs truth")

    colors = cm.rainbow(np.linspace(0,1, len(plevels)))

    minlim = np.min([np.min(Y_pred), np.min(Y)])
    maxlim = np.max([np.max(Y_pred), np.max(Y)])

    for i, plevel in enumerate(plevels):
        y_pred = Y_pred[:,i]
        y = Y[:, i]

        plt.scatter(y, y_pred, color=colors[i], label=f'{plevel} hPa')

    plt.xlabel("MiMA [m/s^2]")
    plt.ylabel("ANN [m/s^2]")
    lims = [minlim, maxlim]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.legend()

    # Make figures full screen
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(os.path.join(save_path, f"aggregate_predictions_vs_truth.png"))
    plt.close(fig)
"""
