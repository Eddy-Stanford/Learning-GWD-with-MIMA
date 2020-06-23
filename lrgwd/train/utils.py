import os
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from lrgwd.models.baseline import BaseLine
from lrgwd.models.config import VALID_MODELS
from lrgwd.train.config import MONITOR_METRIC
from lrgwd.utils.io import from_pickle
from sklearn.utils import shuffle
from tensorflow.keras import utils
from tensorflow.keras.callbacks import \
    CSVLogger  # Check if needed when using TensorBoard
from tensorflow.keras.callbacks import TerminateOnNaN  # Check if needed
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)


# TODO: Rewrite so this does not depend on string literals (i.e. baseline)
def get_model(model: str):
    if model == "baseline":
        return BaseLine()
    else: 
        raise Exception("In valid model. Pick one of: f{VALID_MODELS}")

def get_metadata(source_path: Union[os.PathLike, str]):
    """
    Gets metadata from source_path
    """
    metadata_fn = os.path.join(source_path, "metadata.pkl")
    metadata = from_pickle(metadata_fn)
    return metadata


def get_callbacks(save_path: Union[os.PathLike, str]):
    os.makedirs(os.path.join(save_path, "checkpoints"))
    return [
        EarlyStopping(
            monitor=MONITOR_METRIC, patience=10, restore_best_weights=True,
        ),
        CSVLogger(
            os.path.join(save_path, "training.log"), separator=',', append=False
        ),
        ReduceLROnPlateau(
            monitor=MONITOR_METRIC, factor=0.1, patience=5, verbose=1, mode='min',
        ),
        ModelCheckpoint(
            filepath=os.path.join(save_path, "checkpoints/weights.{epoch:02d}.hdf5"),
            save_best_only=False,
            save_weights_only=True,
            monitor=MONITOR_METRIC,
            mode="min",
            save_freq='epoch', 
        ),
        TensorBoard(log_dir=os.path.join(save_path,"logs")),
        TerminateOnNaN()
    ]


class DataGenerator(utils.Sequence):
    def __init__(
        self, 
        tensors_scaler,
        target_scaler,
        tensors_filepath: Union[os.PathLike, str],
        target_filepath: Union[os.PathLike, str],
        num_samples: int,
        batch_size: int = 32, 
        chunk_size: int = 500, 
    ):
        self.batch_size=batch_size
        self.chunk_size=chunk_size
        self.tensors_filepath = tensors_filepath
        self.target_filepath = target_filepath
        self.num_samples = num_samples
        self.tensors_scaler = tensors_scaler
        self.target_scaler = target_scaler 

        self.generator = self._get_batch()
        

    def __len__(self):
        # Denotes the number of batches per epoch 
        return int(np.floor(self.num_samples) / self.batch_size)

    
    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        X, y = next(self.generator)
        return (X, y)


    def _get_batch(self):
        for tensors_chunk, target_chunk in zip(
            pd.read_csv(self.tensors_filepath, chunksize=self.chunk_size),
            pd.read_csv(self.target_filepath, chunksize=self.chunk_size)
        ):
            # create batch from chunk 
            tensors_chunk, target_chunk = (tensors_chunk.to_numpy(), target_chunk.to_numpy())
            # standardize chunk
            tensors_chunk = self.tensors_scaler.transform(tensors_chunk)
            target_chunk  = self.target_scaler.transform(target_chunk)
            # convert to tf Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((tensors_chunk, target_chunk)).repeat()
            
            train_dataset = train_dataset.shuffle(buffer_size=self.chunk_size).batch(self.batch_size)
            for train_batch, target_batch in train_dataset:
                target_batch = np.hsplit(target_batch, 18)
                yield train_batch, target_batch
