import os
from collections import defaultdict

import click
import numpy as np
from lrgwd.config import FEATURES, NON_ZERO_GWD_PLEVELS
from lrgwd.ingestor.config import DEFAULTS
from lrgwd.ingestor.utils import generate_metrics
from lrgwd.ingestor.visualize import (plot_distribution,
                                      plot_distribution_per_level,
                                      plot_scatter)
from lrgwd.utils.io import to_json
from lrgwd.utils.logger import logger
from lrgwd.utils.tracking import tracking
from scipy.io import netcdf


@click.command("ingestor")
@click.option(
    "--save-path",
    default=DEFAULTS["save_path"],
    show_default=True,
    help="File path to save raw dataset as npz",
)
@click.option(
    "--source-path",
    default=DEFAULTS["source_path"],
    show_default=True,
    help="File path to source data as CDF"
)
@click.option(
    "--convert/--no-convert",
    default=True,
    show_default=True,
    help="Convert cdf fields to npz files. Otherwise, only visualize data",
)
@click.option(
    "--visualize/--no-visualize",
    default=True,
    show_default=True,
    help="Include visual plots",
)
@click.option(
    '--tracking/--no-tracking',
    default=True,
    show_default=True,
    help="Track run using mlflow"
)
@click.option("--verbose/--no-verbose", default=True)
def main(**params):
    with tracking(
        experiment="ingestor",
        params=params,
        local_dir=params['save_path'],
        tracking=params['tracking']
    ): 
        with netcdf.netcdf_file(params['source_path'], 'r') as cdf_data:
            save_path = params["save_path"]
            os.makedirs(save_path, exist_ok=True)

            # Convert CDF File to NPZ File
            if params["convert"]:
                if params['verbose']: 
                    logger.info(f"Converting CDF to NPZ (this may take a few minutes)")

                # Convert from float32 to float16 for all varaibles except gwfu, gwfv and hght to reduce memory usage (extra precision unnecessary)
                feats = defaultdict(str)
                for feat in FEATURES:
                    feat_data = cdf_data.variables[feat][:]
                    if feat not in ["hght", "gwfu_cgwd", "gwfv_cgwd", "vcomp", "ucomp", "omega"]:
                        feat_data = np.float16(feat_data)
                    feats[feat] = feat_data
                np.savez_compressed(
                    os.path.join(save_path, 'raw_data.npz'), 
                    **feats
                )


            # Create histograms and compute useful metrics
            feat_info = {}
            if params["visualize"]:
                if params["verbose"]: logger.info("Visualizing data distributions")
                for feat in FEATURES:
                    if params["verbose"]: logger.info(f"Plot {feat}")
                    feat_info[feat] = generate_metrics(feat, cdf_data.variables[feat])
                    feat_data = cdf_data.variables[feat][:]

                    plot_distribution(
                        feat_info=feat_info[feat], 
                        feat_data=feat_data,
                        save_path=save_path
                    )

                    if feat == "gwfu_cgwd" or feat == "gwfv_cgwd":
                        plevels = cdf_data.variables['level'][:]
                        plot_distribution_per_level(
                            feat_info=feat_info[feat], 
                            feat_data=feat_data,
                            plevels=plevels, 
                            save_path=save_path
                        )
                
                if params["verbose"]: logger.info(f"Plot scatter")
                plot_scatter(
                    X_info=feat_info["gwfv_cgwd"],
                    y_info=feat_info["gwfu_cgwd"],
                    X=cdf_data.variables["gwfv_cgwd"][:][:,:NON_ZERO_GWD_PLEVELS,:,:],
                    y=cdf_data.variables["gwfu_cgwd"][:][:,:NON_ZERO_GWD_PLEVELS,:,:],
                    save_path=save_path,
                )

                to_json(os.path.join(save_path, "feat_info.json"), feat_info) 

if __name__ == "__main__":
    main()
