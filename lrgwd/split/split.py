import os
from typing import Any, Union

import numpy as np
import pandas as pd
from lrgwd.extractor.config import GWFU_FN, GWFV_FN, LABELS_FN, TENSORS_FN
from lrgwd.split.config import (TEST_GWFU_FN, TEST_GWFV_FN, TEST_LABELS_FN,
                                TEST_TENSORS_FN, TRAIN_GWFU_FN, TRAIN_GWFV_FN,
                                TRAIN_LABELS_FN, TRAIN_TENSORS_FN, VAL_GWFU_FN,
                                VAL_GWFV_FN, VAL_LABELS_FN, VAL_TENSORS_FN)
from lrgwd.split.preprocess import Preprocessor
from lrgwd.utils.io import from_pickle, to_pickle


class Splitter():
    def __init__(self, save_path: Union[os.PathLike, str]):
        # Save Paths
        self.train_tensors_path = os.path.join(save_path, TRAIN_TENSORS_FN)
        self.train_labels_path = os.path.join(save_path, TRAIN_LABELS_FN)
        self.train_gwfu_path = os.path.join(save_path, TRAIN_GWFU_FN)
        self.train_gwfv_path = os.path.join(save_path, TRAIN_GWFV_FN)

        self.val_tensors_path = os.path.join(save_path, VAL_TENSORS_FN)
        self.val_labels_path = os.path.join(save_path, VAL_LABELS_FN)
        self.val_gwfu_path = os.path.join(save_path, VAL_GWFU_FN)
        self.val_gwfv_path = os.path.join(save_path, VAL_GWFV_FN)

        self.test_tensors_path = os.path.join(save_path, TEST_TENSORS_FN)
        self.test_labels_path = os.path.join(save_path, TEST_LABELS_FN)
        self.test_gwfu_path = os.path.join(save_path, TEST_GWFU_FN)
        self.test_gwfv_path = os.path.join(save_path, TEST_GWFV_FN)

        self.include_train_header = True
        self.include_val_header = True
        self.include_test_header = True
    

    def save_train(
        self,
        tensors: pd.DataFrame,
        gwfu: pd.DataFrame,
        gwfv: pd.DataFrame,
        labels: pd.DataFrame
    ):
        tensors.to_csv(self.train_tensors_path, mode='a', header=self.include_train_header, index=False)
        gwfu.to_csv(self.train_gwfu_path, mode='a', header=self.include_train_header, index=False)
        gwfv.to_csv(self.train_gwfv_path, mode='a', header=self.include_train_header, index=False)
        labels.to_csv(self.train_labels_path, mode='a', header=self.include_train_header, index=False)
        self.include_train_header = False

    def save_val(
        self,
        tensors: pd.DataFrame,
        gwfu: pd.DataFrame,
        gwfv: pd.DataFrame,
        labels: pd.DataFrame
    ):
        tensors.to_csv(self.val_tensors_path, mode='a', header=self.include_val_header, index=False)
        gwfu.to_csv(self.val_gwfu_path, mode='a', header=self.include_val_header, index=False)
        gwfv.to_csv(self.val_gwfv_path, mode='a', header=self.include_val_header, index=False)
        labels.to_csv(self.val_labels_path, mode='a', header=self.include_val_header, index=False)
        self.include_val_header = False

    def save_test(
        self,
        tensors: pd.DataFrame,
        gwfu: pd.DataFrame,
        gwfv: pd.DataFrame,
        labels: pd.DataFrame
    ):
        tensors.to_csv(self.test_tensors_path, mode='a', header=self.include_test_header, index=False)
        gwfu.to_csv(self.test_gwfu_path, mode='a', header=self.include_test_header, index=False)
        gwfv.to_csv(self.test_gwfv_path, mode='a', header=self.include_test_header, index=False)
        labels.to_csv(self.test_labels_path, mode='a', header=self.include_test_header, index=False)
        self.include_test_header = False

def save_metadata(
    save_path: Union[os.PathLike, str],
    source_path: Union[os.PathLike, str],
    metadata: Any,
):
    prev_metadata = from_pickle(os.path.join(source_path, "metadata.pkl"))
    # Shallow Merge
    metadata = {**prev_metadata, **metadata}
    to_pickle(
        path=os.path.join(save_path, "metadata.pkl"),
        obj=metadata
    )


def split(
    num_samples: int, 
    test_split: float,
    val_split: float,
    save_path: Union[os.PathLike, str],
    source_path: Union[os.PathLike, str],
    cnn_features: bool,
    batch_size: int,
): 
    # Source Paths
    tensors_path = os.path.join(source_path, TENSORS_FN)    
    gwfu_path = os.path.join(source_path, GWFU_FN)
    gwfv_path = os.path.join(source_path, GWFV_FN)
    labels_path = os.path.join(source_path, LABELS_FN)

    splitter = Splitter(save_path)
    preprocessor = Preprocessor(save_path)

    # Number of samples in each set
    num_test_samples = np.floor(test_split*num_samples)
    num_val_samples = np.floor(val_split*num_samples)
    num_train_samples = num_samples - (num_test_samples + num_val_samples)
    save_metadata(
        source_path=source_path, 
        save_path=save_path,
        metadata={
            "num_test_samples": num_test_samples,
            "num_val_samples": num_val_samples,
            "num_train_samples": num_train_samples,
        }
    )

    batch_size = min([num_test_samples, num_val_samples, batch_size])

    num_read = 0
    for tensor_chunk, gwfu_chunk, gwfv_chunk, labels_chunk in zip(
        pd.read_csv(tensors_path, chunksize=batch_size),
        pd.read_csv(gwfu_path, chunksize=batch_size),
        pd.read_csv(gwfv_path, chunksize=batch_size),
        pd.read_csv(labels_path, chunksize=batch_size),
    ):
        # TODO: Consider shuffling chunk and presplitting. Then saving to all three split
        # Doing so would make split datasets more representative of entire dataset

        if num_read < num_train_samples:
            splitter.save_train(
                tensors=tensor_chunk,
                gwfu=gwfu_chunk, 
                gwfv=gwfv_chunk, 
                labels=labels_chunk
            )
            preprocessor.partial_fit(
                tensors=tensor_chunk,
                gwfu=gwfu_chunk, 
                gwfv=gwfv_chunk, 
            )
        elif num_train_samples <= num_read and num_read < (num_train_samples + num_val_samples):
            splitter.save_val(
                tensors=tensor_chunk,
                gwfu=gwfu_chunk, 
                gwfv=gwfv_chunk, 
                labels=labels_chunk
            )
        else: 
            splitter.save_test(
                tensors=tensor_chunk,
                gwfu=gwfu_chunk, 
                gwfv=gwfv_chunk, 
                labels=labels_chunk
            )
        
        num_read += batch_size

    preprocessor.save()
