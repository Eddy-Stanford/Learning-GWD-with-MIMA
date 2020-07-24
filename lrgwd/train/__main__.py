import os

import click
from lrgwd.train.config import DEFAULTS
from lrgwd.train.utils import (DataGenerator, get_callbacks, get_metadata,
                               get_model)
from lrgwd.utils.io import from_pickle
from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking

"""
Trains the model outlined in baseline.
"""
@click.command("train")
@click.option(
    "--save-path",
    default=DEFAULTS["save_path"],
    show_default=True,
    type=str,
    help="File path to save trained model",
)
@click.option(
    "--source-path",
    default=DEFAULTS["source_path"],
    multiple=True,
    show_default=True,
    type=str,
    help="File path to split datasets",
)
@click.option(
    "--model",
    default=DEFAULTS["model"],
    show_default=True,
    type=str,
    help="Model to train. Defaults to BaseLine. Must be a model from models/"
)
@click.option(
    "--batch-size",
    default=DEFAULTS["batch_size"],
    show_default=True,
    type=int,
    help="Size of training batch"
)
@click.option(
    "--chunk-size",
    default=DEFAULTS["chunk_size"],
    show_default=True,
    type=int,
    help="Size of chunk loaded into memory as once",
)
@click.option(
    "--epochs",
    default=DEFAULTS["epochs"],
    show_default=True,
    type=int,
    help="Num epochs to train"
)
@click.option(
    "--target",
    default=DEFAULTS["target"],
    show_default=True,
    help="Either gwfu or gwfv"
)
@click.option(
    "--steps-per-epoch",
    default=DEFAULTS["steps_per_epoch"],
    show_default=True,
    help="Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch."
)
@click.option(
    "--validation-steps",
    default=DEFAULTS["validation_steps"],
    show_default=True,
    help="Total number of steps (batches of samples) to draw before stopping when performing validation at the end of every epoch. \
        If 'validation_steps' is None, validation will run until the validation_data dataset is exhausted. \
        In the case of an infinitely repeated dataset, it will run into an infinite loop",
)
@click.option(
    "--tracking/--no-tracking",
    default=True,
    show_default=True,
    help="Track run using mlflow"
)
@click.option(
    "--num-workers",
    default=DEFAULTS["num_workers"],
    show_default=True,
    help="Only use if multiprocessing is True"
)
@click.option(
    "--learning-rate",
    default=DEFAULTS["learning_rate"],
    show_default=True,
)
@click.option(
    "--scaler-path",
    help="Path to Standard scaler",
)
@click.option("--use-multiprocessing/--no-use-multiprocessing", default=True)
@click.option("--verbose/--no-verbose", default=True)
def main(**params):
    """
    Train Model
    """
    with tracking(
        experiment="train",
        params=params,
        local_dir=params["save_path"],
        tracking=params["tracking"]
    ):
        target = params["target"]
        os.makedirs(params["save_path"], exist_ok=True)
        metadata = get_metadata(params["source_path"][0])

        # Get Model
        Model = get_model(params["model"])
        model = Model.build((metadata["input_shape"],), metadata["output_shape"], params["learning_rate"])

        # Get scalers
        tensors_scaler = from_pickle(os.path.join(params["scaler_path"], "tensors_scaler.pkl"))
        target_scaler = from_pickle(os.path.join(params["scaler_path"], f"{target}_scaler.pkl"))


        # Create data generators
        train_generator = DataGenerator(
            tensors_filepath=[os.path.join(path, "train_tensors.csv") for path in params["source_path"]],
            target_filepath=[os.path.join(path, "../full_features", f"train_{target}.csv") for path in params["source_path"]],
            batch_size=params["batch_size"],
            chunk_size=params["chunk_size"],
            num_samples=metadata["total_samples"]*len(params["source_path"]),
            tensors_scaler=tensors_scaler,
            target_scaler=target_scaler,
            name="train",
        )

        val_generator = DataGenerator(
            tensors_filepath=[os.path.join(path, "val_tensors.csv") for path in params["source_path"]],
            target_filepath=[os.path.join(path, "../full_features", f"val_{target}.csv") for path in params["source_path"]],
            batch_size=params["batch_size"],
            chunk_size=params["chunk_size"],
            num_samples=metadata["total_samples"]*len(params["source_path"]),
            tensors_scaler=tensors_scaler,
            target_scaler=target_scaler,
            name="val",
        )

        # Fit Model
        callbacks = get_callbacks(params["save_path"], params["model"])
        # model.run_eagerly = True
        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            steps_per_epoch=params["steps_per_epoch"],
            validation_steps=params["validation_steps"],
            epochs=params["epochs"],
            verbose=params["verbose"],
            callbacks=callbacks,
            use_multiprocessing=params["use_multiprocessing"],
        )


if __name__ == "__main__":
    main()
