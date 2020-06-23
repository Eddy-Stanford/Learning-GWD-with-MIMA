import os

import click

from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking
from lrgwd.utils.io import from_pickle
from lrgwd.train.config import DEFAULTS
from lrgwd.train.utils import get_model, get_metadata, get_callbacks, DataGenerator


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
        metadata = get_metadata(params["source_path"])

        # Get Model
        Model = get_model(params["model"])
        model = Model.build(metadata["input_shape"], metadata["output_shape"])

        # Get scalers
        tensors_scaler = from_pickle(os.path.join(params["source_path"], "tensors_scaler.pkl")) 
        target_scaler = from_pickle(os.path.join(params["source_path"], f"{target}_scaler.pkl")) 
        

        # Create data generators
        train_generator = DataGenerator(
            tensors_filepath=os.path.join(params["source_path"], "train_tensors.csv"),
            target_filepath=os.path.join(params["source_path"], f"train_{target}.csv"),
            batch_size=params["batch_size"],
            chunk_size=params["chunk_size"],
            num_samples=metadata["total_samples"],
            tensors_scaler=tensors_scaler,
            target_scaler=target_scaler,
        )

        val_generator = DataGenerator(
            tensors_filepath=os.path.join(params["source_path"], "val_tensors.csv"),
            target_filepath=os.path.join(params["source_path"], f"val_{target}.csv"),
            batch_size=params["batch_size"],
            chunk_size=params["chunk_size"],
            num_samples=metadata["total_samples"],
            tensors_scaler=tensors_scaler,
            target_scaler=target_scaler,
        )

        # Fit Model
        callbacks = get_callbacks(params["save_path"])
        history = model.fit(
            x=train_generator,
            validation_data=val_generator,
            epochs=params["epochs"],
            verbose=params["verbose"],
            callbacks=callbacks,
            use_multiprocessing=params["use_multiprocessing"],
        )


if __name__ == "__main__":
    main()
