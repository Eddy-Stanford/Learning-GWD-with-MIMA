import csv
import os
from typing import Union

import click
import numpy as np
import pandas as pd
from lrgwd.extractor.config import TENSORS_FN
from lrgwd.split.config import DEFAULTS
from lrgwd.split.split import split
from lrgwd.utils.io import from_pickle
from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking


@click.command("split")
@click.option(
    "--save-path",
    default=DEFAULTS["save_path"],
    show_default=True,
    type=str,
    help="File path to save extracted dataset",
)
@click.option(
    "--source-path",
    default=DEFAULTS["source_path"],
    show_default=True,
    type=str,
    help="File path to raw dataset as npz",
)
@click.option(
    "--using-cnn-features",
    default=False,
    show_default=True,
    help="Using 3d feature matricies",
)
@click.option(
    "--test-split",
    default=DEFAULTS["test_split"],
    show_default=True,
    help="Fraction between 0.0 and 1.0 of data to reserve for testing"
)
@click.option(
    "--val-split",
    default=DEFAULTS["val_split"],
    show_default=True,
    help="Fraction between 0.0 and 1.0 of data to reserve for validation"
)
@click.option(
    "--preprocess/--no-preprocess", 
    default=True,
    show_default=True,
    help="Standardize and Normalize data"
)
@click.option(
    "--scaling-factor",
    default=1.0,
    show_default=True,
    help="How much to scale gwfu and gwfv values by. Defaults to 1.0 (i.e. identity)"
)
@click.option(
    "--num-samples",
    default=None,
    show_default=True,
    help="Number of samples to take from extracted vectors. If None, use all"
)
@click.option(
    "--batch-size",
    default=DEFAULTS["batch_size"],
    show_default=True,
    type=int,
    help="Number of feature vectors to process before writing"
)
@click.option(
    "--tracking/--no-tracking",
    default=True,
    show_default=True,
    help="Track run using mlflow"
)
@click.option(
    "--scaler-path", 
    default=DEFAULTS["save_path"],
)
@click.option("--load-scaler/--no-load-scaler", default=False)
@click.option("--verbose/--no-verbose", default=True)
def main(**params):
    with tracking(
       experiment="split",
       params=params,
       local_dir=params["save_path"],
       tracking=params["tracking"]
    ):
        os.makedirs(params["save_path"], exist_ok=True)

        if params["num_samples"] is None: 
            num_samples = int(get_num_samples(params["source_path"]))
        else: 
            num_samples = int(params["num_samples"])

        if params["verbose"]:
            logger.debug(f"Splitting {num_samples} samples")

        split(
            num_samples=num_samples,
            test_split=params["test_split"],
            val_split=params["val_split"],
            save_path=params["save_path"],
            source_path=params["source_path"],
            cnn_features=params["using_cnn_features"],
            batch_size=params["batch_size"],
            scaler_info={"load": params["load_scaler"], "path": params["scaler_path"]},
        )


def get_num_samples(
    source_path: Union[str, os.PathLike]
) -> int:
    """
    Finds the number of samples in source_path/tensors.csv
    Subtract one to remove header from count
    """
    metadata = from_pickle(os.path.join(source_path, "metadata.pkl"))
    return metadata["total_samples"]
