import os
from typing import Dict, List, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle, to_pickle
from lrgwd.utils.logger import logger
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error


def plot_predictions_vs_truth_per_level(
    predictions: np.ndarray, 
    targets: np.ndarray,
    title: str,
    color: str,
    save_path: Union[os.PathLike, str],
): 
    minlim = np.min([np.min(predictions), np.min(targets)])
    maxlim = np.max([np.max(targets), np.max(predictions)])

    fig = plt.figure()
    _ = plt.axes(aspect="equal")
    plt.scatter(targets, predictions, color=color)
    plt.xlabel("True Values [m/s^2]")
    plt.ylabel("Predictions [m/s^2]")
    lims = [minlim, maxlim]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    # Make figures full screen
    fig.set_size_inches(32, 18)
    fig.savefig(os.path.join(save_path, f"{title}_predictions_vs_truth.png"))
    plt.close(fig)


def plot_predictions_vs_truth(
    raw_predictions: np.ndarray,
    raw_targets: np.ndarray,
    test_predictions: Dict[str, np.ndarray], 
    test_targets: Dict[str, np.ndarray],
    save_path: Union[os.PathLike, str],
) -> None:
    num_plevels = raw_predictions[0].shape[0]
    colors = cm.rainbow(np.linspace(0,1, num_plevels)) 

    for i, values in enumerate(zip(test_predictions.values(), test_targets.values())):
        predictions, targets = values
        plot_predictions_vs_truth_per_level(
            predictions=predictions, 
            targets=targets,
            title=str(i),
            color=colors[i],
            save_path=save_path,
        )

    fig = plt.figure()
    _ = plt.axes(aspect="equal")

    minlim = np.min([np.min(raw_predictions), np.min(raw_targets)])
    maxlim = np.max([np.max(raw_predictions), np.max(raw_targets)])

    for i, values in enumerate(zip(test_predictions.values(), test_targets.values())):
        predictions, targets = values
        plt.scatter(targets, predictions, color=colors[i], label=f'plevel_{i}')

    plt.xlabel("True Values [m/s^2]")
    plt.ylabel("Predictions [m/s^2]")
    lims = [minlim, maxlim]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.legend()
    
    # Make figures full screen
    fig.set_size_inches(32, 18)
    fig.savefig(os.path.join(save_path, f"aggregate_predictions_vs_truth.png"))
    plt.close(fig)


def plot_distributions_per_level(
    test_predictions: Dict[str, np.ndarray],
    test_targets: Dict[str, np.ndarray], 
    save_path: Union[os.PathLike, str],
) -> None: 
    # Iterate through each pressure level
    for i, values in enumerate(zip(test_predictions.values(), test_targets.values())):
        predictions, targets = values

        fig = plt.figure(figsize=(8,6))
        bins = 1000
        plt.hist(predictions, bins, alpha=0.5, label='predictions', density=True)
        plt.hist(targets, bins, alpha=0.5, label='targets', density=True)
        # plt.hist(
        #     [predictions, targets], 
        #     bins=bins, 
        #     density=True, 
        #     label=["predictions", "targets"]
        # )

        plt.xlabel("gwfu (m/s^2)", size=14)
        plt.ylabel("Count", size=14)
        plt.title(f"Histogram Predictions vs Targets for Plevel {i}")
        plt.legend(loc='upper right')

        # Make figures full screen
        fig.set_size_inches(32, 18)
        fig.savefig(os.path.join(save_path, f"predictions_targets_histogram_{i}.png"))
        plt.close(fig)


def plot_r_squared(
    r_squared: List[float],
    save_path: Union[os.PathLike, str],
) -> None: 
    fig = plt.figure(figsize=(8,6))
    plevels = list(range(len(r_squared)))
    plt.plot(plevels, r_squared)
    plt.ylabel("r squared", size=14)
    plt.xlabel("PLevels", size=14)
    plt.title(f"R squared vs Plevels")
    plt.xticks(plevels)
    plt.yticks(np.arange(0,1.0,.1))

    # Make figures full screen
    fig.set_size_inches(32, 18)
    fig.savefig(os.path.join(save_path, f"r_squared.png"))
    plt.close(fig)



# def plot_metrics(metrics: Dict[str, np.ndarray], save_path: Union[os.PathLike, str]):
#     num_plevels = metrics["maes"].shape[0]
#     fig = plt.figure(figsize=(num_plevels, 6))
#     use_labels = True
#     colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
#     for i in range(num_plevels):
#         plevel_metrics = {
#             "mae": metrics["maes"][i],
#             "std": metrics["maes"][i],
#             "min": metrics["mins"][i],
#             "max": metrics["maxes"][i],
#             "mean": metrics["means"][i],
#             "median": metrics["medians"][i],
#         }
#         ax = plt.subplot(num_plevels, 1, 1+i)
#         plot_numberline(
#             metrics=plevel_metrics,
#             ylabel=f"{i}",
#             colors=colors,
#             ax=ax,
#             use_labels=use_labels
#         )
#         use_labels = False

    
#     fig.legend()
#     fig.savefig(os.path.join(save_path, "metrics.png"))
#     plt.close(fig)


# def plot_numberline(
#     metrics: Dict[str, float], ylabel: str, colors: List[str], ax, use_labels: bool
# ) -> None:
#     setup_lineplot(ax, xlim=(metrics["min"], metrics["max"]))
#     ax.xaxis.set_major_locator(ticker.AutoLocator())
#     ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
#     ax.set_ylabel(ylabel)

#     x = list(metrics.values())
#     y = [0]*len(x)
#     labels = [None]*len(x)
#     if use_labels:
#         labels = list(metrics.keys())
#     for i in range(len(x)):
#         ax.scatter(x[i], y[i], c=colors[i], label=labels[i])

# # Setup a plot such that only the bottom spine is shown
# def setup_lineplot(ax, xlim):
#     ax.spines['right'].set_color('none')
#     ax.spines['left'].set_color('none')
#     ax.yaxis.set_major_locator(ticker.NullLocator())
#     ax.spines['top'].set_color('none')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.tick_params(which='major', width=1.00)
#     ax.tick_params(which='major', length=5)
#     ax.tick_params(which='minor', width=0.75)
#     ax.tick_params(which='minor', length=2.5)
#     ax.set_xlim(xlim[0], xlim[1])
#     ax.set_ylim(0, 1)
#     ax.patch.set_alpha(0.0)
