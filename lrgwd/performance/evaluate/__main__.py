import glob
import os
from typing import Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lrgwd.performance.config import DEFAULTS
from lrgwd.performance.evaluate.utils import (EvaluationPackage,
                                              generate_metrics)
from lrgwd.performance.visualize import (plot_distributions_per_level,
                                         plot_predictions_vs_truth,
                                         plot_r_squared)
from lrgwd.train.utils import get_model
from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle
from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking
from tensorflow import keras

"""
Evaluates the given model using test_tensors.csv from source-path. Creates predicted_vs_true diagrams,
histograms, and r^2 values for each pressure level. This command also saves the full set of predictions and
target values for future analysis. To evaluate using a subsamples of `test_tensors.csv` using set --num-test-samples.
To remove outliers according to ther z-score, set --remove-outliers

Example Usage:
python lrgwd evaluate \
    --save-path ./runs/models/LogCosh/evaluate \
    --source-path ./runs/data/four_years/split \
    --model-path ./runs/models/LogCosh/baseline.100.h5 \
    --remove-outliers 3.5 \
    --num-test-samples 5000000
"""
@click.command("evaluate")
@click.option(
    "--model-path",
    default=DEFAULTS["model_path"],
    show_default=True,
    type=str,
    help="Filepath to model"
)
@click.option(
    "--save-path",
    default=DEFAULTS["evaluate"]["save_path"],
    show_default=True,
    type=str,
    help="File path to save evaluation plots"
)
@click.option(
    "--scaler-path",
    default=DEFAULTS["evaluate"]["source_path"],
    show_default=True,
    type=str,
    help="File path to Standard Scaler"
)
@click.option(
    "--source-path",
    default=DEFAULTS["evaluate"]["source_path"],
    show_default=True,
    type=str,
    help="Path to labels and test data"
)
@click.option(
    "--remove-outliers",
    default=None,
    show_default=True,
    help="Removes outliers with z-score threshold. If None, do not remove outliers"
)
@click.option(
    "--num-test-samples",
    default=None,
    show_default=True,
    help="Number of samples to test with. If None, use the whole test dataset"
)
@click.option(
    "--target",
    default=DEFAULTS["target"],
    show_default=True,
    help="Either gwfu or gwfv",
)
@click.option(
    "--tracking/--no-tracking",
    default=True,
    show_default=True,
    help="Track run using mlflow"
)
@click.option("--verbose/--no-verbose", default=True)
@click.option("--visualize/--no-visualize", default=True)
def main(**params):
    """
    Evaluate Model
    """
    with tracking(
        experiment="evaluate",
        params=params,
        local_dir=params["save_path"],
        tracking=params["tracking"],
    ):
        os.makedirs(params["save_path"], exist_ok=True)

        # Load Model
        if params["verbose"]: logger.info("Loading Model")
        model = keras.models.load_model(os.path.join(params["model_path"]))
        model.summary()

        # Load Test Data
        if params["verbose"]: logger.info("Loading Data and Making Predictions")
        evaluation_package = EvaluationPackage(
            source_path=params["source_path"],
            scaler_path=params["scaler_path"],
            num_samples=params["num_test_samples"],
            target=params["target"],
            remove_outliers=params["remove_outliers"],
            save_path=params["save_path"],
            model=model,
        )

        # Visualize and Metrics
        if params["verbose"]: logger.info("Generate Metrics")
        metrics = generate_metrics(
            targets=evaluation_package.targets,
            predictions=evaluation_package.predictions,
            plevel_predictions=evaluation_package.plevel_predictions,
            plevel_targets=evaluation_package.plevel_targets,
            save_path=params["save_path"],
        )

        if params["visualize"]:
            if params["verbose"]: logger.info("Visualize")
            plot_distributions_per_level(
                plevel_targets=evaluation_package.plevel_targets,
                plevel_predictions=evaluation_package.plevel_predictions,
                save_path=params["save_path"]
            )

            plot_predictions_vs_truth(
                predictions=evaluation_package.predictions,
                targets=evaluation_package.targets,
                plevel_predictions=evaluation_package.plevel_predictions,
                plevel_targets=evaluation_package.plevel_targets,
                save_path=params["save_path"],
            )

            plot_r_squared(
                r_squared=metrics["r_squared"],
                save_path=params["save_path"]
            )
