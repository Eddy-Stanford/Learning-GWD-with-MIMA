import csv
import logging
import os

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DEFAULT = {
    "test_split": .10,
    "validation_split": .10,
    "window": (1440, 12, 64, 128)
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNKSIZE = 2

@click.command("split")
@click.option(
    "--split-data-path",
    default="./data/mini/split/",
    show_default=True,
    help="File path to save split and processed dataset",
)
@click.option(
    "--extracted-data-path",
    default="./data/mini/extracted/",
    show_default=True,
    help="File path to previously extracted dataset",
)
@click.option(
    "--test-split",
    default=DEFAULT["test_split"],
    show_default=True,
    help="Fraction between 0.0 and 1.0 of data to reserve for testing"
)
@click.option(
    "--validation-split",
    default=DEFAULT["validation_split"],
    show_default=True,
    help="Fraction between 0.0 and 1.0 of data to reserve for validation"
)
@click.option(
    "--window/--no-window",
    default=True,
    show_default=True,
    help="Use window to select subset of data to use."
)
@click.option(
    "--window-dims",
    default=DEFAULT["window"],
    show_default=True,
    help="Window in (1440, 22, 64, 128) to use. Defaults to top ten pressure levels"
)
def main(**params):
    """
    TODO:
        create dirs
        split - using window or full
        scale 
    """
    os.makedirs(params["split_data_path"], exist_ok=True)
    fvectors_path = os.path.join(params["extracted_data_path"], "feature_vectors.csv")
    gwfu_labels_path = os.path.join(params["extracted_data_path"], "gwfu_labels.csv")
    gwfv_labels_path = os.path.join(params["extracted_data_path"], "gwfv_labels.csv")

    with open(fvectors_path, "r") as fvectors_csv, \
         open(gwfu_labels_path, "r") as gwfu_labels_csv, \
         open(gwfv_labels_path, "r") as gwfv_labels_csv:

        fvectors_reader = csv.reader(fvectors_csv)
        gwfu_reader = csv.reader(gwfu_labels_csv)
        gwfv_reader = csv.reader(gwfv_labels_csv)

        num_fvectors = get_num_fvectors(fvectors_path)
        split(
            split_path=params["split_data_path"],
            fvectors_reader=fvectors_reader,
            gwfu_reader=gwfu_reader,
            gwfv_reader=gwfv_reader,
            num_fvectors=num_fvectors,
            val_split=params["validation_split"],
            test_split=params["test_split"]
        )

def split(
    split_path: os.PathLike,
    fvectors_reader: csv.reader,
    gwfu_reader: csv.reader,
    gwfv_reader: csv.reader, 
    num_fvectors: int,
    val_split: float,
    test_split: float
) -> None:
    # Calculate number of each cohort
    num_val =  int(num_fvectors*val_split)
    num_test = int(num_fvectors*test_split)
    num_train = num_fvectors - num_val - num_test
    
    # Create file paths
    train_path = os.path.join(split_path, "train_data.csv")
    train_labels_path = os.path.join(split_path, "train_labels.csv")

    val_path = os.path.join(split_path, "val_data.csv")
    val_labels_path = os.path.join(split_path, "val_labels.csv")
    
    test_path = os.path.join(split_path, "test_data.csv")
    test_labels_path = os.path.join(split_path, "test_labels.csv")

    on_train, on_val, on_test = (True, False, False)
    scaler_fvectors = StandardScaler()
    scaler_gwfu = StandardScaler()
    scaler_gwfv = StandardScaler()
    with open(train_path,"w+") as train_f, \
         open(val_path,"w+") as val_f, \
         open(test_path,"w+") as test_f, \
         open(train_labels_path, "w+") as train_labels_f, \
         open(val_labels_path, "w+") as val_labels_f, \
         open(test_labels_path, "w+") as test_labels_f:
        # create_writers
        train_writer = csv.writer(train_f)
        val_writer = csv.writer(val_f)
        test_writer = csv.writer(test_f)
        train_labels_writer = csv.writer(train_labels_f)
        val_labels_writer = csv.writer(val_labels_f)
        test_labels_writer = csv.writer(test_labels_f)

        # Writer Headers
        fvectors_header = next(fvectors_reader)
        train_writer.writerow(fvectors_header)
        val_writer.writerow(fvectors_header)
        test_writer.writerow(fvectors_header)

        # Use next to skip headers
        next(gwfu_reader)
        next(gwfv_reader)

        count = 0
        for fvector, gwfu_label, gwfv_label in tqdm(zip(
            fvectors_reader, gwfu_reader, gwfv_reader
        ), "splitting data"):
            # Reshape data so that StandardScaler see it is one sample, rather than one feature
            fvector = np.reshape(fvector, (1, -1))
            gwfu_label = np.reshape(gwfu_label, (1, -1))
            gwfv_label = np.reshape(gwfv_label, (1, -1))

            if count == num_train and on_train:
                on_train, on_val = (False, True)
                count = 0

            elif count == num_val and on_val:
                on_val, on_test = (False, True)

            if on_train:
                # Fit scaler
                scaler_fvectors.partial_fit(fvector)
                scaler_gwfu.partial_fit(gwfu_label)
                scaler_gwfv.partial_fit(gwfv_label)
                # Write
                train_writer.writerow(fvector)            
                train_labels_writer.writerow(gwfu_label)

            elif on_test:
                # Transform
                transformed_fvector = scaler_fvectors.transform(fvector)
                transformed_gwfu_label = scaler_gwfu.transform(gwfu_label)
                # Write 
                test_writer.writerow(transformed_fvector)            
                test_labels_writer.writerow(transformed_gwfu_label)

            elif on_val:
                # Transform
                transformed_fvector = scaler_fvectors.transform(fvector)
                transformed_gwfu_label = scaler_gwfu.transform(gwfu_label)
                # Write
                val_writer.writerow(transformed_fvector)
                val_labels_writer.writerow(transformed_gwfu_label)

            else: 
                logging.warning("Feature vector not captured")
            
            count += 1

    return {
        "scalers": {
            "gwfu": scaler_gwfu,
            "gwfv": scaler_gwfv,
            "fvectors": scaler_fvectors,
        },
        "train_paths": {
            "labels": train_labels_path,
            "fvectors": train_path
        }
    }

def get_num_fvectors(
    extracted_path: os.PathLike
) -> float:
    fvectors_object = csv.reader(extracted_path)
    num_fvectors = sum(1 for fvector in fvectors_object)
    num_fvectors -=1 # exclude header in count

    return num_fvectors
