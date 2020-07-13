import os
from typing import Dict, List, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from tqdm import tqdm 
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error

from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle, to_pickle
from lrgwd.utils.logger import logger


class EvaluationPackage(object):
    def __init__(self, 
        source_path: Union[os.PathLike, str],
        num_samples: Union[None, float],
        target: str,
        remove_outliers: Union[str, float],
        save_path: Union[os.PathLike, str],
        model,
    ) -> None:

        test_tensors_fp = os.path.join(source_path, "train_tensors.csv")
        test_targets_fp = os.path.join(source_path, f"train_{target}.csv")

        # Get Scalers
        tensors_scaler_fp = os.path.join(source_path, "tensors_scaler.pkl")
        tensors_scaler = from_pickle(tensors_scaler_fp)

        target_scaler_fp = os.path.join(source_path, f"{target}_scaler.pkl")
        target_scaler = from_pickle(target_scaler_fp)

        self.predictions = []
        self.targets = []
        chunksize = 1000000
        num_total_predictions = 0
        if num_samples < chunksize: chunksize = num_samples

        for test_tensors, test_targets in tqdm(zip(
            pd.read_csv(test_tensors_fp, chunksize=chunksize), 
            pd.read_csv(test_targets_fp, chunksize=chunksize), 
        ), "Load test data"):
            if num_samples is not None and num_total_predictions >= num_samples: break 

            test_tensors = test_tensors.to_numpy()
            test_targets = test_targets.to_numpy()

            # Transform Targets
            test_tensors = tensors_scaler.transform(test_tensors)

            self.targets.append(test_targets)
            self.predictions.append(
                self.predict(
                    model=model, 
                    tensors=test_tensors,
                    target_scaler=target_scaler,
                )
            )
        
            num_total_predictions += chunksize

        self.predictions = np.concatenate(np.array(self.predictions), axis=0)
        self.targets = np.concatenate(np.array(self.targets), axis=0)


        # Removes outliers and returns dictionary keyed on each pressure level
        self.plevel_predictions, self.plevel_targets = self.split_predictions_on_plevel(
            predictions=self.predictions, 
            targets=self.targets,
            outliers=remove_outliers,
        )

        # Save unaltered predictions and targets
        to_pickle(
            path=os.path.join(save_path, "predictions.pkl"),
            obj={
                "predictions": self.predictions,
                "targets": self.targets,
            }
        )        


    def predict(self, model, tensors, target_scaler):
        predictions = model.predict(tensors)
        predictions = np.hstack(predictions)
        predictions = target_scaler.inverse_transform(predictions)

        return predictions

    
    def split_predictions_on_plevel(self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        outliers: Union[None, float]
    ):
        # Split predictions per level 
        plevel_predictions = {}
        plevel_targets = {}

        num_plevels = predictions.shape[1]
        for i in range(num_plevels):
            slice_predictions = predictions[:, i]
            slice_targets = targets[:, i]

            # Remove Outliers
            if outliers is not None:
                plevel_predictions, plevel_targets = self.remove_outliers(
                    predictions=slice_predictions,
                    targets=slice_targets,
                    outliers=float(outliers),
                )
            
            plevel_predictions[f"plevel_{i}"] = slice_predictions
            plevel_targets[f"plevel_{i}"] = slice_targets
        
        return plevel_predictions, plevel_targets


    def remove_outliers(self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        outliers: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_outliers = is_outlier(targets, thresh=outliers)
        predictions = predictions[~target_outliers]
        targets = targets[~target_outliers]

        prediction_outliers = is_outlier(predictions, thresh=outliers)
        predictions = predictions[~prediction_outliers]
        targets = targets[~prediction_outliers]

        return (predictions, targets)


def generate_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray,
    plevel_predictions: Dict[str, np.ndarray],
    plevel_targets: Dict[str, np.ndarray],
    save_path: Union[os.PathLike, str]
) -> None:
    # Pressure Level Specific Metrics
    metrics = {
        "maes": mean_absolute_error(targets, predictions, multioutput="raw_values"),
        "rmse": mean_squared_error(targets, predictions, multioutput="raw_values", squared=False),
        "stds": np.std(targets, axis=1),
        "mins": np.min(targets, axis=1),
        "maxes": np.max(targets, axis=1),
        "means": np.mean(targets, axis=1),
        "medians": np.median(targets, axis=1),
    }

    metrics["r_squared"] = calculate_r_squared(
        test_predictions=plevel_predictions, 
        test_targets=plevel_targets,
    )

    to_pickle(os.path.join(save_path, "metrics.pkl"), metrics)

    return metrics


def calculate_r_squared(
    test_predictions: Dict[str, np.ndarray],
    test_targets: Dict[str, np.ndarray],
) -> List[float]:
    r_squared = []
    for i, values in enumerate(zip(test_predictions.values(), test_targets.values())):
        predictions, targets = values
        slope, intercept, r_value, p_value, std_err = linregress(predictions, targets)
        r_squared.append(r_value**2)
    return r_squared

