import os

import click
from tensorflow import keras

from lrgwd.evaluate.config import DEFAULTS
from lrgwd.evaluate.utils import load_weights, generate_model
from lrgwd.train.utils import get_model
from lrgwd.utils.tracking import tracking


@click.command("evaluate")
@click.option(
    "--weights-path",
    default=DEFAULTS["weights_path"],
    show_default=True,
    type=str,
    help="File path to model weights from training"
)
@click.option(
    "--model",
    default=DEFAULTS["model"],
    show_default=True,
    type=str,
    help="Name of model"
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
        model = keras.models.load_model(
            os.path.join(params["weights_path"], "weights.10.hdf5")
        )
        model.summary()

        
        # model = load_weights(params["model_path"])
        # generate_model()
