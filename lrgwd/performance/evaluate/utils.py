import os
from typing import Dict, List, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle, to_pickle
from lrgwd.utils.logger import logger
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error


class EvaluationPackage(object):
    def __init__(self, test_tensors, test_targets, tensors_scaler, target_scaler):
        self.test_tensors = test_tensors
        self.test_targets = test_targets
        self.test_labels = None
        self.tensors_scaler = tensors_scaler
        self.target_scaler = target_scaler 

    def predict(self, model, outliers, save_path):
        test_predictions = model.predict(self.test_tensors)
        test_predictions = np.hstack(test_predictions)
        self.raw_test_predictions = self.target_scaler.inverse_transform(test_predictions)

        # Split predictions per level 
        num_plevels = self.raw_test_predictions[0].shape[0]
        self.predictions = {}
        self.targets = {}
        for i in range(num_plevels):
            plevel_predictions = self.raw_test_predictions[:, i]
            plevel_targets = self.test_targets[:, i]

            # Remove Outliers
            if outliers is not None:
                plevel_predictions, plevel_targets = self.remove_outliers(
                    predictions=plevel_predictions,
                    targets=plevel_targets,
                    outliers=float(outliers),
                )

            self.predictions[f"plevel_{i}"] = plevel_predictions
            self.targets[f"plevel_{i}"] = plevel_targets

        to_pickle(
            path=os.path.join(save_path, "predictions.pkl"),
            obj={
                "predictions": self.predictions,
                "targets": self.targets,
            }
        )        

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


def generate_evaluation_package(
    source_path: Union[os.PathLike, str],
    num_samples: Union[None, float],
    target: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_tensors_fp = os.path.join(source_path, "test_tensors.csv")
    test_targets_fp = os.path.join(source_path, f"test_{target}.csv")


    if num_samples is None: 
        test_targets = pd.read_csv(test_targets_fp).to_numpy()
        test_tensors = pd.read_csv(test_tensors_fp).to_numpy()
    else: 
        for test_tensors, test_targets in zip(
            pd.read_csv(test_tensors_fp, chunksize=num_samples), 
            pd.read_csv(test_targets_fp, chunksize=num_samples), 
        ):
            test_tensors = test_tensors.to_numpy()
            test_targets = test_targets.to_numpy()
            break

    # Transform Targets
    tensors_scaler_fp = os.path.join(source_path, "tensors_scaler.pkl")
    tensors_scaler = from_pickle(tensors_scaler_fp)
    test_tensors = tensors_scaler.transform(test_tensors)

    target_scaler_fp = os.path.join(source_path, f"{target}_scaler.pkl")
    target_scaler = from_pickle(target_scaler_fp)


    return EvaluationPackage(
        test_tensors=test_tensors,
        test_targets=test_targets,
        tensors_scaler=tensors_scaler,
        target_scaler=target_scaler,
    )


def generate_metrics(
    test_predictions: np.ndarray, 
    test_targets: np.ndarray,
    plevel_test_predictions: Dict[str, np.ndarray],
    plevel_test_targets: Dict[str, np.ndarray],
    save_path: Union[os.PathLike, str]
) -> None:
    # Pressure Level Specific Metrics
    metrics = {
        "maes": mean_absolute_error(test_targets, test_predictions, multioutput="raw_values"),
        "rmse": mean_squared_error(test_targets, test_predictions, multioutput="raw_values", squared=False),
        "stds": np.std(test_targets, axis=1),
        "mins": np.min(test_targets, axis=1),
        "maxes": np.max(test_targets, axis=1),
        "means": np.mean(test_targets, axis=1),
        "medians": np.median(test_targets, axis=1),
    }

    metrics["r_squared"] = calculate_r_squared(
        test_predictions=plevel_test_predictions, 
        test_targets=plevel_test_targets,
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
