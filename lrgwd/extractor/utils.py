import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from lrgwd.config import (NON_ZERO_GWD_PLEVELS, TARGET_FEATURES,
                          TRAIN_FEATURES, VERTICAL_COLUMN_FEATURES)
from lrgwd.extractor.config import GWFU_FN, GWFV_FN, LABELS_FN, TENSORS_FN
from lrgwd.utils.io import to_pickle
from lrgwd.utils.logger import logger
from tqdm import tqdm


class Data(object):
    def __init__(self, data):
        self.npz_data = data
        self.data_shape = data["gwfu_cgwd"].shape
        self.time = self.data_shape[0]
        self.plevels = self.data_shape[1]
        self.lat = self.data_shape[2]
        self.lon = self.data_shape[3]

def save_metadata(
    path: os.PathLike,
    total_samples: int,
    features: List[str],
    plevels: int,
    output_shape: Tuple[int],
    indx: List[int],
) -> None:
    """
    Pickles Metadata:
        input_shape: shape of train tensor for vertical column
        output_shape: shape of gwfu or gwfv tensor for vertical column
        indx: indicies that we used to shuffle
        total samples extracted
    """
    input_shape = len(features)
    num_vc_feat = 0
    for vc_feat in VERTICAL_COLUMN_FEATURES:
        if vc_feat in features:
            input_shape -= 1
            num_vc_feat += 1

    input_shape = input_shape*plevels + num_vc_feat
    to_pickle(
        path=path,
        obj={
            "total_samples": total_samples,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "indx": indx,
        }
    )


def vectorized_extract_tensors(
    data: Data,
    save_path: Union[os.PathLike, str],
    num_samples: Union[int, None],
    features: List[str],
    plevels: int,
    verbose: bool = True,
    shuffle: bool = True,
) -> None:
    """
    NOTE: This is far more efficient than `extract_tensors` and solves previous speed issues,
    but individually saves each feature into its own file. After they are pasted together to
    create `tensors.csv` be sure to delete each feature file to avoid unnecessary memory usage.

    Extracts feature tensors and target columns from raw data.

    Arguments:
    ----------
    data (Data): npz data
    save_path (Union[os.PathLike, str]): path to save all files
    num_samples (Union[int, None]): number of samples to extract from data.
        If None, extracts all samples.
    plevels (int): number of pressure levels to include in tensors.
        Use to ignore low altitude pressure levels
    verbose (bool)
    shuffle (bool)
    """
    npz_data = data.npz_data
    total_num_samples = data.time*data.lat*data.lon

    if num_samples == None or num_samples > total_num_samples:
        logger.warning(f"Extracting all possible samples {total_num_samples}")
        num_samples = total_num_samples

    idx = np.random.choice(np.arange(total_num_samples), num_samples, replace=False)
    # Save Metadata
    save_metadata(
        path=os.path.join(save_path, "metadata.pkl"),
        total_samples=num_samples,
        features=features,
        plevels=plevels,
        output_shape=(NON_ZERO_GWD_PLEVELS,),
        indx=idx,
    )

    features.extend(TARGET_FEATURES)
    paste_command = f"paste "
    for feat in features:
        if feat == "lat":
            feat_data = np.indices((data.time, data.lat, data.lon))[1,:,:,:]
        elif feat == "lon":
            feat_data = np.indices((data.time, data.lat, data.lon))[2,:,:,:]
        else:
            # Memory Load Feature
            if verbose: logger.info(f"Memory Loading {feat}")
            feat_data = np.array(npz_data[feat])

        # Extract Feature
        if verbose: logger.info(f"Extracting {feat}")
        if feat in VERTICAL_COLUMN_FEATURES: #== "slp":
            # Flatten time, lat, lon dimensions
            feat_data = np.stack(feat_data)
            feat_data = feat_data.reshape(1, total_num_samples)
        else:
            if feat in TARGET_FEATURES:
                # Flatten time, lat, lon dimensions while preserving vertical columns
                feat_data = np.stack(feat_data[:,:NON_ZERO_GWD_PLEVELS,:,:], axis=1)
                feat_data = feat_data.reshape(NON_ZERO_GWD_PLEVELS, total_num_samples)
            elif feat in TRAIN_FEATURES:
                feat_data = np.stack(feat_data[:,:plevels,:,:], axis=1)
                feat_data = feat_data.reshape(plevels, total_num_samples)

        # Sample & Shuffle
        if shuffle: feat_data = feat_data[:, idx]
        feat_data = feat_data.T
        if feat in TRAIN_FEATURES:
            feat_label = feat
            paste_command += os.path.join(save_path, f"{feat}.csv ")
        elif feat in TARGET_FEATURES:
            if feat == "gwfu_cgwd":
                feat_label = "gwfu"
            else:
                feat_label = "gwfv"
        else:
            logger.warning("Unused attribute")
            continue

        pd.DataFrame(feat_data).to_csv(os.path.join(save_path, f"{feat_label}.csv"), mode="a+", index=False, header=False)

    logger.info(f"Pasting files together {paste_command}")
    os.system(paste_command + " -d ',' > " + os.path.join(save_path, "tensors.csv"))
    os.system("paste gwfu.csv gwfv.csv" + " -d ',' > " + os.path.join(save_path, "combined.csv"))



######################### KEEP FOR REFERENCE ##############################
######################### LEGACY CODE #####################################
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
######################### KEEP FOR REFERENCE ##############################
######################### LEGACY CODE #####################################


def extract_3D_tensors(
    data: Data,
    save_path: Union[os.PathLike, str],
    step_size: int,
    num_steps: int,
    start_time: int,
    num_samples: Union[int, None],
) -> None:
    """
    TODO: ONLY IMPLEMENT IF USING 1D TENSORS UNDERPERFORMS
    """
    pass
