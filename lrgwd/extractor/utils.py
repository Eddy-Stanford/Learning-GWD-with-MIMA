import os
from typing import Dict, Union

import numpy as np
import pandas as pd
from lrgwd.config import (NON_ZERO_GWD_PLEVELS, TARGET_FEATURES, TENSOR,
                          TRAIN_FEATURES)
from lrgwd.extractor.config import GWFU_FN, GWFV_FN, LABELS_FN, TENSORS_FN
from lrgwd.utils.io import to_pickle
from lrgwd.utils.logger import logger
from tqdm import tqdm


class Data(object):
    def __init__(self, data):
        self.raw_data = data
        self.data_shape = data["gwfu_cgwd"].shape
        self.time = self.data_shape[0]
        self.plevels = self.data_shape[1]
        self.lat = self.data_shape[2]
        self.lon = self.data_shape[3]


def extract_tensors(
    data: Data,
    save_path: Union[os.PathLike, str],
    num_samples: Union[int, None],
    plevels: int,
    batch_size: int,
) -> None:
    """
    Extracts feature tensors and target columns from raw data.

    Arguments:
    ----------
    data (Data): 
    save_path (Union[os.PathLike, str]): path to save all files
    num_samples (Union[int, None]): number of samples to extract from data. 
        If None, extracts all samples.
    plevels (int): number of pressure levels to include in tensors. 
        Use to ignore low altitude pressure levels
    batch_size (int): number of samples to gather before writing to disk. Useful for
        environment with more memory or in time termination.
    
    Returns:
    --------
    None
    """
    raw_data = data.raw_data

    # If num_samples not set, default to using all data
    max_samples = data.time*data.lat*data.lon 
    if not num_samples or num_samples > max_samples: 
        logger.warning("Extracting all possible samples")
        num_samples = max_samples

    first_batch = True
    tensors, tensors_labels = (pd.DataFrame(), pd.DataFrame())
    targets_gwfu, targets_gwfv = (pd.DataFrame(), pd.DataFrame())
    for i in tqdm(range(num_samples), "Extracting Tensors"):
        t, lat, lon = np.unravel_index(i, (data.time, data.lat, data.lon))
        tensor, tensor_labels = (pd.DataFrame(), pd.DataFrame())

        for feat in TENSOR:
            if feat == "slp":
                # Labels
                labels = pd.DataFrame([f"slp_{t}_{lat}_{lon}"])
                tensor_labels = pd.concat([tensor_labels, labels], copy=False)

                # Vertical Column
                slp = pd.DataFrame(data=[raw_data[feat][t, lat, lon]])
                tensor = pd.concat([tensor, slp], copy=False)
            else:  
                if feat in TRAIN_FEATURES:
                    # Labels
                    labels = pd.DataFrame([
                        f"{feat}_{t}_{plevel}_{lat}_{lon}" 
                        for plevel in range(0, plevels)
                    ])
                    vertical_column = pd.DataFrame(data=raw_data[feat][t, :plevels, lat, lon])

                    tensor = pd.concat([tensor, vertical_column], copy=False)
                    tensor_labels = pd.concat([tensor_labels, labels], copy=False)
                elif feat in TARGET_FEATURES: 
                    # Vertical Column
                    vertical_column = pd.DataFrame(
                        data=raw_data[feat][t, :NON_ZERO_GWD_PLEVELS, lat, lon],
                    )
                    if feat == "gwfu_cgwd":
                        targets_gwfu = pd.concat([targets_gwfu, vertical_column], axis=1) 
                    else: 
                        targets_gwfv = pd.concat([targets_gwfv, vertical_column], axis=1) 
                else: 
                    logger.warning("Unused attribute")

                
        # Concat tensors to batch
        tensors = pd.concat([tensors, tensor], axis=1) 
        tensors_labels = pd.concat([tensors_labels, tensor_labels], axis=1) 

        if tensors.shape[1] == batch_size:
            save_batch(
                tensors=tensors, 
                labels=tensors_labels, 
                targets_gwfu=targets_gwfu,
                targets_gwfv=targets_gwfv,
                save_path=save_path, 
                include_header=first_batch
            ) 
            if first_batch:
                to_pickle(
                    path=os.path.join(save_path, "metadata.pkl"),
                    obj={
                        "total_samples": num_samples,
                        "input_shape": tensors.iloc[:,0].shape,
                        "output_shape": targets_gwfu.iloc[:,0].shape,
                    }
                )
            first_batch=False
            tensors, tensors_labels = (pd.DataFrame(), pd.DataFrame())
            targets_gwfu, targets_gwfv = (pd.DataFrame(), pd.DataFrame())
    
            
def save_batch(
    tensors: pd.DataFrame,
    labels: pd.DataFrame,
    targets_gwfu: pd.DataFrame,
    targets_gwfv: pd.DataFrame,
    save_path: Union[os.PathLike, str],
    include_header: bool,
) -> None:
    """
    Write batch to save_path
    """
    tensors_path = os.path.join(save_path, TENSORS_FN) 
    gwfu_path = os.path.join(save_path, GWFU_FN) 
    gwfv_path = os.path.join(save_path, GWFV_FN) 
    labels_path = os.path.join(save_path, LABELS_FN) 

    tensors.T.to_csv(tensors_path, mode='a', header=include_header, index=False)
    targets_gwfu.T.to_csv(gwfu_path, mode='a', header=include_header, index=False)
    targets_gwfv.T.to_csv(gwfv_path, mode='a', header=include_header, index=False)
    labels.T.to_csv(labels_path, mode="a", header=include_header, index=False)



def extract_3D_tensors(
    data: Data, 
    save_path: Union[os.PathLike, str],
    step_size: int, 
    num_steps: int,
    start_time: int,
    num_samples: Union[int, None],
) -> None: 
    """
    TODO: IMPLEMENT
    """
    pass
