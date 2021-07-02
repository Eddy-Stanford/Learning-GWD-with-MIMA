import os
import random
from typing import Any, List, Tuple, Union
import traceback
import threading

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.utils import shuffle
from tensorflow.keras import utils
from tensorflow.keras.callbacks import \
    CSVLogger  # Check if needed when using TensorBoard
from tensorflow.keras.callbacks import TerminateOnNaN  # Check if needed
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)

from lrgwd.config import NON_ZERO_GWD_PLEVELS
from lrgwd.models.baseline import BaseLine, compile_model
from lrgwd.models.wavenet_2 import Wavenet_2
from lrgwd.models.config import VALID_MODELS
from lrgwd.train.config import MONITOR_METRIC
from lrgwd.utils.io import from_pickle
from lrgwd.utils.logger import logger


def optimize_model(
        model,
        save_path: Union[os.PathLike, str],
        model_name: str,
        train_generator,
        val_generator,
        steps_per_epoch,
        val_steps,
        epochs,
        end_step,
        learning_rate = .001,
    ):
    """
    Perform pruning and quantization on a pre-trained model:
    https://www.tensorflow.org/model_optimization/guide

    model: tf.keras model
    """

    # Begin Pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                 final_sparsity=0.50,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model = compile_model(model_for_pruning, learning_rate)
    logger.info("Begin Pruning")
    model_for_pruning.summary()

    callbacks = get_pruning_callbacks(save_path, model_name)

    model_for_pruning.fit(
        x=train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks
    )

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    # Save hdf5
    fp = os.path.join(save_path, model_name)
    tf.keras.models.save_model(model_for_export, fp + '_final.hdf5')

    # Begin Quantization
    logger.info("Begin Pruning")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    # Save Tflite
    with open(fp + '.tflite', 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)



def get_pruning_callbacks(save_path: Union[os.PathLike, str], model_name: str = "baseline"):
    return [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=os.path.join(save_path, 'logs_pruning')),
        EarlyStopping(
            monitor=MONITOR_METRIC, patience=10, restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor=MONITOR_METRIC, factor=0.1, patience=5, verbose=1, mode='min',
        ),
        ModelCheckpoint(
            filepath=os.path.join(save_path, f"{model_name}" + ".{epoch:02d}.hdf5"),
            save_best_only=True,
            monitor=MONITOR_METRIC,
            mode="min",
        ),
    ]


# TODO: Rewrite so this does not depend on string literals (i.e. baseline)
def get_model(model: str):
    if model == "baseline":
        return BaseLine()
    elif model == "wavenet_2":
        return Wavenet_2()
    else:
        raise Exception(f"Invalid model. Pick one of: {VALID_MODELS}")


def get_metadata(source_path: Union[os.PathLike, str]):
    """
    Gets metadata from source_path
    """
    metadata_fn = os.path.join(source_path, "metadata.pkl")
    metadata = from_pickle(metadata_fn)
    return metadata


def load_model(model_path: str, learning_rate: float):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model = compile_model(model, learning_rate)
        return model
    except:
        traceback.print_exc()
        logger.error(f"Failed to load model: {model_path}")


def get_callbacks(save_path: Union[os.PathLike, str], model_name: str = "baseline"):
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
            filepath=os.path.join(save_path, f"{model_name}" + ".{epoch:02d}.hdf5"),
            save_best_only=True,
            monitor=MONITOR_METRIC,
            mode="min",
        ),
        TensorBoard(log_dir=os.path.join(save_path,"logs")),
        TerminateOnNaN()
    ]


class DataGenerator(utils.Sequence):
    def __init__(
        self,
        tensors_scaler,
        target_scaler,
        tensors_filepath: List[Union[os.PathLike, str]],
        target_filepath: List[Union[os.PathLike, str]],
        num_samples: int,
        name: str,
        batch_size: int = 32,
        chunk_size: int = 500,
        train_with_random: bool = False,
        num_outputs: int = 33,
    ):
        self.batch_size=batch_size
        self.chunk_size=chunk_size
        self.tensors_filepath = tensors_filepath
        self.target_filepath = target_filepath
        self.num_samples = num_samples
        self.tensors_scaler = tensors_scaler
        self.target_scaler = target_scaler

        self.name = name
        self.num_files = len(self.tensors_filepath)
        self.train_with_random = train_with_random
        self.generators = self._create_generators()
        self.lock = threading.Lock()
        self.batch_num = 0
        self.num_outputs = num_outputs


    def _create_generators(self):
        generators = {}
        for i, fps in enumerate(zip(self.tensors_filepath, self.target_filepath)):
            tensors_fp, target_fp = fps
            batch_size = int(np.floor(self.batch_size / self.num_files))
            # if i == (self.num_files - 1): batch_size += 1

            generators[(tensors_fp, target_fp, batch_size)] = self._get_batch(tensors_fp, target_fp, batch_size, self.train_with_random)


        return generators

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(self.num_samples) / self.batch_size)


    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        while True:
            with self.lock:
                X, y = [], []
                for key, gen in self.generators.items():
                    tensors_fp, target_fp, batch_size = key
                    try:
                        X_gen, y_gen = next(gen)
                    except StopIteration:
                        gen = self._get_batch(tensors_fp, target_fp, batch_size, self.train_with_random)
                        self.generators[key] = gen
                        X_gen, y_gen = next(gen)
                    X.append(X_gen)

                    if True:
                        y.append(y_gen)
                    else:
                        for i in range(len(y_gen)):
                            if len(y) > i:
                                y[i] = np.concatenate([y[i], y_gen[i]], axis=0)
                            else:
                                y.append(y_gen[i])
                X = np.concatenate(X, axis=0)
                y = np.concatenate(y, axis=0)

                return (X, y)


    def _get_batch(self, tensors_filepath, target_filepath, batch_size, train_with_random):
        for tensors_chunk, target_chunk in zip(
            pd.read_csv(tensors_filepath, header=0, chunksize=self.chunk_size),
            pd.read_csv(target_filepath, header=0, chunksize=self.chunk_size)
        ):
            # create batch from chunk
            tensors_chunk, target_chunk = (tensors_chunk.to_numpy(), target_chunk.to_numpy())
            # standardize chunk
            tensors_chunk = self.tensors_scaler.transform(tensors_chunk)
            target_chunk  = self.target_scaler.transform(target_chunk)

            # convert to tf Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((tensors_chunk, target_chunk))

            train_dataset = train_dataset.shuffle(buffer_size=self.chunk_size).batch(batch_size, drop_remainder=True)
            for train_batch, target_batch in train_dataset:
                #target_batch = np.hsplit(target_batch, self.num_outputs)
                yield train_batch, target_batch
