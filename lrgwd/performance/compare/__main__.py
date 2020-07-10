import glob
import os
from typing import Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lrgwd.performance.config import DEFAULTS
from lrgwd.train.utils import get_model
from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle
from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking
from tensorflow import keras


@click.command("compare")
@click.option(
    "--save-path",
    default=DEFAULTS["compare"]["save_path"],
    show_default=True,
    type=str,
    help="File path to save evaluation plots"
)
@click.option(
    "--source-path",
    default=DEFAULTS["compare"]["source_path"],
    show_default=True,
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
@click.option("--visualize/--no-visualize", default=True)
def main(**params):
    """
    Evaluate Model
    """
    with tracking(
        experiment="compare",
        params=params,
        local_dir=params["save_path"],
        tracking=params["tracking"],
    ):
        os.makedirs(params["save_path"], exist_ok=True)

        logger.info("Compare models")
