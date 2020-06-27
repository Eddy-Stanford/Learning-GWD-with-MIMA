import glob
import os
from typing import Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lrgwd.evaluate.config import DEFAULTS
from lrgwd.evaluate.utils import (generate_evaluation_package,
                                  generate_metrics,
                                  plot_distributions_per_level,
                                  plot_predictions_vs_truth)
from lrgwd.train.utils import get_model
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
    default=DEFAULTS["save_path"],
    show_default=True,
    type=str,
    help="File path to save evaluation plots"
)
@click.option(
    "--source-path",
    default=DEFAULTS["source_path"],
    show_default=True,
    type=str,
    help="Path to labels and test data"
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

        if params["verbose"]: logger.info("Loading Data")
        # Load Test Data
        evaluation_package = generate_evaluation_package(
            source_path=params["source_path"],
            target=params["target"],
        )

        # Predict
        if params["verbose"]: logger.info("Generate Predictions")
        test_predictions = model.predict(evaluation_package.test_tensors)
        test_predictions = np.hstack(test_predictions)
        test_predictions = evaluation_package.target_scaler.inverse_transform(test_predictions)

        if params["verbose"]: logger.info("Visualize")
        metrics = generate_metrics(
            test_predictions=test_predictions, 
            test_targets=evaluation_package.test_targets,
            test_labels=evaluation_package.test_labels,
            save_path=params["save_path"],
        )

        plot_distributions_per_level(
            metrics=metrics,
            test_predictions=test_predictions,
            test_targets=evaluation_package.test_targets,
            save_path=params["save_path"]
        )

        plot_predictions_vs_truth(
            test_predictions=test_predictions,
            test_targets=evaluation_package.test_targets,
            save_path=params["save_path"],
        )
