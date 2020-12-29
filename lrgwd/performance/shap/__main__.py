import glob
import os
from typing import Union

import click
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lrgwd.performance.config import DEFAULTS
from lrgwd.performance.shap.utils import (EvaluationPackage,
                                          calculate_rsquared,
                                          generate_metrics,
                                          plot_predictions_vs_truth)
from lrgwd.train.utils import get_model
from lrgwd.utils.data_operations import is_outlier
from lrgwd.utils.io import from_pickle
from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking
from tensorflow import keras

"""
Shap evaluates the model using test_tensors.csv from source-path. Creates predicted_vs_true diagram,
sums total drag, and computes shap values. To evaluate using a subsample of `test_tensors.csv` using set --num-test-samples.
To remove outliers according to their z-score, set --remove-outliers

Example Usage:
python lrgwd shap \
    --save-path ./runs/models/LogCosh/evaluate \
    --source-path ./runs/data/four_years/split \
    --model-path ./runs/models/LogCosh/baseline.100.h5 \
    --remove-outliers 3.5 \
    --num-test-samples 5000000
"""
@click.command("shap")
@click.option(
    "--model-path",
    default=DEFAULTS["model_path"],
    show_default=True,
    type=str,
    help="Filepath to model"
)
@click.option(
    "--save-path",
    default=DEFAULTS["evaluate"]["save_path"],
    show_default=True,
    type=str,
    help="File path to save evaluation plots"
)
@click.option(
    "--scaler-path",
    default=DEFAULTS["evaluate"]["source_path"],
    show_default=True,
    type=str,
    help="File path to Standard Scaler"
)
@click.option(
    "--source-path",
    default=DEFAULTS["evaluate"]["source_path"],
    show_default=True,
    type=str,
    help="Path to labels and test data"
)
@click.option(
    "--remove-outliers",
    default=None,
    show_default=True,
    help="Removes outliers with z-score threshold. If None, do not remove outliers"
)
@click.option(
    "--num-test-samples",
    default=None,
    show_default=True,
    help="Number of samples to test with. If None, use the whole test dataset"
)
@click.option(
    "--target",
    default=DEFAULTS["target"],
    show_default=True,
    help="Either gwfu or gwfv",
)
@click.option(
    "--tracking/--no-tracking",
    default=True,
    show_default=True,
    help="Track run using mlflow"
)
@click.option(
    "--evaluate-with-random/--no-evaluate-with-random",
    default=False,
    show_default=True,
    help="Evaluate using randomly generated tensors (loc=0.0, scale=1.0)"
)
@click.option(
    "--plevel",
    default=2,
    show_default=True,
    help="Pressure Level to perform shap analysis for"
)
@click.option("--verbose/--no-verbose", default=True)
@click.option("--visualize/--no-visualize", default=True)
def main(**params):
    """
    Generate Shap Values
    """
    with tracking(
        experiment="shap",
        params=params,
        local_dir=params["save_path"],
        tracking=params["tracking"],
    ):
        os.makedirs(params["save_path"], exist_ok=True)

        # Load Model
        if params["verbose"]: logger.info("Loading Model")
        print(params["model_path"])
        model = keras.models.load_model(os.path.join(params["model_path"]), compile=False)
        model.summary()

        # Load Mini Data
        if params["verbose"]: logger.info("Loading Data and Making Predictions")
        evaluation_package = EvaluationPackage(
            source_path=params["source_path"],
            scaler_path=params["scaler_path"],
            num_samples=int(params["num_test_samples"]),
            target=params["target"],
            save_path=params["save_path"],
            model=model,
        )

        plevels = [1.80e-01, 5.60e-01, 7.20e-01, 9.40e-01,
                   1.21e+00, 1.57e+00, 2.02e+00, 2.60e+00,
                   3.32e+00, 4.25e+00, 5.40e+00, 6.85e+00,
                   8.68e+00, 1.09e+01, 1.38e+01, 1.73e+01,
                   2.16e+01, 2.68e+01, 3.32e+01, 4.11e+01,
                   5.07e+01, 6.22e+01, 7.60e+01, 9.24e+01,
                   1.12e+02, 1.35e+02, 1.62e+02, 1.94e+02,
                   2.31e+02, 2.73e+02, 3.21e+02, 3.75e+02,
                   4.36e+02, 5.03e+02, 5.77e+02, 6.55e+02,
                   7.37e+02, 8.21e+02, 9.02e+02, 9.71e+02]

        if params["verbose"]: logger.info("Calculate R Squared")
        r_squared = calculate_rsquared(
            Y_pred=evaluation_package.Y_pred,
            Y=evaluation_package.Y_raw,
            plevels=plevels[:33],
        )

        # Visualize and Metrics
        plot_predictions_vs_truth(
            Y_pred=evaluation_package.Y_pred,
            Y=evaluation_package.Y_raw,
            plevels=plevels[:33],
            r_squared=r_squared,
            save_path="/scratch/zespinos/Learning-GWD-with-MIMA",
        )
        return

        ##### CREATE FEATURE NAMES VECTOR #####
        #features = ["T", "H", "u", "v", "w"]
        features = ["u", "v"]
        feature_names = []
        plevels = [int(p) for p in plevels]
        for feat in features:
            for i, level in enumerate(plevels):
                dif = 13 - i
                feature_names.append(feat + f" {dif}")
        #feature_names.extend(["slp", "lat", "lon"])
        ########################################

        background = 4500
        num_test = 5000

        X = evaluation_package.X

        for pidx, plevel in enumerate(plevels):
            #if pidx >= 33: return
            #if pidx >= 27 or pidx <= 10: continue
            if pidx != 13: continue # or pidx != 13: continue

            pmodel = keras.Model(model.input, model.output[pidx])
            pmodel.compile(loss="logcosh", optimizer="adam")
            pmodel.summary()

            Y = evaluation_package.Y[:, pidx]

            print("PLevel: ", pidx, " Pressure: ", plevel)
            print("X shape: ", evaluation_package.X.shape)
            print("Y shape: ", evaluation_package.Y.shape)

            X_e = X[:background, :]
            X_s = X[background:background+num_test, :]

            e = shap.DeepExplainer(pmodel, X_e)
            print("Created DeepExplainer")
            shap_values = e.shap_values(X_s)

            print("Generated shap values")
            print(shap_values)

            #u_shap = shap_values[0][:,80:120]
            v_shap = shap_values[0][:,80:160]
            #u_X_s = X_s[:,80:120]
            v_X_s = X_s[:,80:160]

            fig = shap.summary_plot(v_shap, v_X_s, max_display=7, feature_names=feature_names, plot_type='bar') #sort=False, show=False)
            print("Created figure")
            ax = plt.gca()
            plt.text(.9, .1, "d)", fontsize=24, transform=ax.transAxes)
            plt.xlabel('mean(|SHAP value|)')
            plt.title(f"Top Wind Predictors of Zonal GWD at 10hPa")
            plt.savefig(f'lrgwd/performance/shap/bar_zonal_predictions_wind_summary_{pidx}.png')
            plt.savefig(f'lrgwd/performance/shap/bar_zonal_predictions_wind_summary_{pidx}.pdf')
            plt.clf()

