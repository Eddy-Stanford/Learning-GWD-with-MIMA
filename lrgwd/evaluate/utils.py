import os
from typing import Dict, List, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lrgwd.utils.io import from_pickle, to_pickle
from sklearn.metrics import mean_absolute_error

from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.logger import logger

class EvaluationPackage(object):
    def __init__(self, test_tensors, test_targets, tensors_scaler, target_scaler):
        self.test_tensors = test_tensors
        self.test_targets = test_targets
        self.test_labels = None
        self.tensors_scaler = tensors_scaler
        self.target_scaler = target_scaler 

    def predict(self, model, outliers):
        test_predictions = model.predict(self.test_tensors)
        test_predictions = np.hstack(test_predictions)
        self.raw_test_predictions = self.target_scaler.inverse_transform(test_predictions)

        # Split predictions per level 
        num_plevels = self.raw_test_predictions[0].shape[0]
        self.predictions = {}
        self.targets = {}
        for i in range(num_plevels):
            plevel_predictions = self.raw_test_predictions[:, i]
            plevel_targets = self.test_targets[:, i]

            # Remove Outliers
            if outliers is not None:
                plevel_predictions, plevel_targets = self.remove_outliers(
                    predictions=plevel_predictions,
                    targets=plevel_targets,
                    outliers=float(outliers),
                )

            self.predictions[f"plevel_{i}"] = plevel_predictions
            self.targets[f"plevel_{i}"] = plevel_targets


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


def generate_evaluation_package(
    source_path: Union[os.PathLike, str],
    num_samples: Union[None, float],
    target: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_tensors_fp = os.path.join(source_path, "train_tensors.csv")
    test_targets_fp = os.path.join(source_path, f"train_{target}.csv")


    if num_samples is None: 
        test_targets = pd.read_csv(test_targets_fp).to_numpy()
        test_tensors = pd.read_csv(test_tensors_fp).to_numpy()
    else: 
        for test_tensors, test_targets in zip(
            pd.read_csv(test_tensors_fp, chunksize=num_samples), 
            pd.read_csv(test_targets_fp, chunksize=num_samples), 
        ):
            test_tensors = test_tensors.to_numpy()
            test_targets = test_targets.to_numpy()
            break

    # Transform Targets
    tensors_scaler_fp = os.path.join(source_path, "tensors_scaler.pkl")
    tensors_scaler = from_pickle(tensors_scaler_fp)
    test_tensors = tensors_scaler.transform(test_tensors)

    target_scaler_fp = os.path.join(source_path, f"{target}_scaler.pkl")
    target_scaler = from_pickle(target_scaler_fp)


    return EvaluationPackage(
        test_tensors=test_tensors,
        test_targets=test_targets,
        tensors_scaler=tensors_scaler,
        target_scaler=target_scaler,
    )


def generate_metrics(
    test_predictions: np.ndarray, 
    test_targets: np.ndarray,
    save_path: Union[os.PathLike, str]
) -> None:
    # Pressure Level Specific Metrics
    metrics = {
        "maes": mean_absolute_error(test_targets, test_predictions, multioutput="raw_values"),
        "stds": np.std(test_targets, axis=1),
        "mins": np.min(test_targets, axis=1),
        "maxes": np.max(test_targets, axis=1),
        "means": np.mean(test_targets, axis=1),
        "medians": np.median(test_targets, axis=1),
    }

    plot_metrics(
        metrics=metrics,
        save_path=save_path,
    )

    to_pickle(os.path.join(save_path, "metrics.pkl"), metrics)

    return metrics


def plot_metrics(metrics: Dict[str, np.ndarray], save_path: Union[os.PathLike, str]):
    num_plevels = metrics["maes"].shape[0]
    fig = plt.figure(figsize=(num_plevels, 6))
    use_labels = True
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    for i in range(num_plevels):
        plevel_metrics = {
            "mae": metrics["maes"][i],
            "std": metrics["maes"][i],
            "min": metrics["mins"][i],
            "max": metrics["maxes"][i],
            "mean": metrics["means"][i],
            "median": metrics["medians"][i],
        }
        ax = plt.subplot(num_plevels, 1, 1+i)
        plot_numberline(
            metrics=plevel_metrics,
            ylabel=f"{i}",
            colors=colors,
            ax=ax,
            use_labels=use_labels
        )
        use_labels = False

    
    fig.legend()
    fig.savefig(os.path.join(save_path, "metrics.png"))
    plt.close(fig)


def plot_numberline(
    metrics: Dict[str, float], ylabel: str, colors: List[str], ax, use_labels: bool
) -> None:
    setup_lineplot(ax, xlim=(metrics["min"], metrics["max"]))
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_ylabel(ylabel)

    x = list(metrics.values())
    y = [0]*len(x)
    labels = [None]*len(x)
    if use_labels:
        labels = list(metrics.keys())
    for i in range(len(x)):
        ax.scatter(x[i], y[i], c=colors[i], label=labels[i])


# Setup a plot such that only the bottom spine is shown
def setup_lineplot(ax, xlim):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)


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
        plt.hist(
            [predictions, targets], 
            bins=1000, 
            label=["predictions", "targets"]
        )
        plt.xlabel("gwfu (m/s^2)", size=14)
        plt.ylabel("Count", size=14)
        plt.title(f"Histogram Predictions vs Targets for Plevel {i}")
        plt.legend(loc='upper right')

        # Make figures full screen
        fig.set_size_inches(32, 18)
        fig.savefig(os.path.join(save_path, f"predictions_targets_histogram_{i}.png"))
        plt.close(fig)
