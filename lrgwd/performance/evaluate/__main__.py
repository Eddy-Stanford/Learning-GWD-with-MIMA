import glob
import os
from typing import Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lrgwd.performance.config import DEFAULTS
from lrgwd.performance.evaluate.utils import (generate_evaluation_package,
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
    default=100000,
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
        if params["verbose"]: logger.info("Loading Data")
        evaluation_package = generate_evaluation_package(
            source_path=params["source_path"],
            num_samples=params["num_test_samples"],
            target=params["target"],
        )

        # Predict
        if params["verbose"]: logger.info("Generate Predictions")
        evaluation_package.predict(
            model, params["remove_outliers"], save_path=params["save_path"]
        )


        # Visualize and Metrics
        if params["verbose"]: logger.info("Generate Metrics")
        metrics = generate_metrics(
            test_targets=evaluation_package.test_targets,
            test_predictions=evaluation_package.raw_test_predictions,
            plevel_test_predictions=evaluation_package.predictions,
            plevel_test_targets=evaluation_package.targets,
            save_path=params["save_path"],
        )

        if params["visualize"]:
            if params["verbose"]: logger.info("Visualize")
            plot_distributions_per_level(
                test_targets=evaluation_package.targets,
                test_predictions=evaluation_package.predictions,
                save_path=params["save_path"]
            )

            plot_predictions_vs_truth(
                raw_predictions=evaluation_package.raw_test_predictions,
                raw_targets=evaluation_package.test_targets,
                test_predictions=evaluation_package.predictions,
                test_targets=evaluation_package.targets,
                save_path=params["save_path"],
            )

            plot_r_squared(
                r_squared=metrics["r_squared"],
                save_path=params["save_path"]
            )
