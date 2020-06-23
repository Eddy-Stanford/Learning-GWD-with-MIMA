import os
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lrgwd.ingestor.config import NUM_SAMPLES_FOR_SCATTER
from lrgwd.utils.data_operations import is_outlier
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter, PercentFormatter


def plot_scatter(
    X_info: Dict[str, Any],
    y_info: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    save_path: Union[os.PathLike, str],
) -> None:
    sns.set(style="white", color_codes=True)

    X_name, X_units = (X_info["name"], X_info["units"])
    y_name, y_units = (y_info["name"], y_info["units"])

    X = X.flatten()
    y = y.flatten()
    idx = np.random.choice(np.arange(len(X)), NUM_SAMPLES_FOR_SCATTER, replace=False)
    X = X[idx]
    y = y[idx]

    fig = sns.jointplot(x=X, y=y)
    fig.set_axis_labels(f"{X_name} ({X_units})", f"{y_name} ({y_units})")
    fig.savefig(os.path.join(save_path, f"{X_name}_vs_{y_name}_scatter.png"))
        

    
def plot_distribution(
    feat_info: Dict[str, Any],
    feat_data: np.ndarray,
    save_path: Union[os.PathLike, str],
) -> None:
    feat_name = feat_info["name"]
    feat_unit = feat_info["units"]
    mu = feat_info["mu"]

    # flatten and remove outliers
    flat_data = feat_data.flatten()
    flat_data = flat_data[~is_outlier(flat_data)]

    # Cut the window in 2 parts
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
    # Add a graph in each part
    sns.boxplot(flat_data, ax=ax_box)
    N, bins, patches = ax_hist.hist(flat_data, bins=100, density=True)

    # Color Histogram bins by height
    fracs = N / N.max()
    # we need to normalize the data to 0.1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
 
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    ax_box.set_title(f"{feat_name}: mu={mu}")
    ax_hist.set(xlabel=f"{feat_unit}", ylabel='%')

    f.savefig(os.path.join(save_path, f"{feat_name}_dist.png"))
 

def plot_distribution_per_level(
    feat_info: Dict[str, Any],
    feat_data: np.ndarray,
    plevels: np.ndarray, 
    save_path: Union[os.PathLike, str],
) -> None:
    long_name = feat_info["long_name"]
    feat_name = feat_info["name"]

    dims = int(np.ceil(np.sqrt(len(plevels))))
    fig, axs = plt.subplots(dims,dims)
    plt.tight_layout()

    for i, plevel in enumerate(plevels):
        plevel = np.format_float_scientific(plevel, precision=1)
        title = f"{plevel}"
        create_histogram(
            raw_data_attribute=feat_data[:,i,:,:],
            axs=axs,
            dims=dims,
            title=title,
            sub_axs=i,
        )

    # Make figures full screen
    fig.set_size_inches(32, 18)

    histogram_path = os.path.join(save_path, f"{feat_name}_histograms.png")
    fig.savefig(histogram_path, bbox_inches='tight')


def create_histogram(
    raw_data_attribute: np.ndarray, 
    axs, #AxsSubPlot
    dims: int,
    title: str,
    sub_axs: int,
) -> None:
    flat_data = raw_data_attribute.flatten()
    flat_data = flat_data[~is_outlier(flat_data)]

    # Metrics
    mu = np.format_float_scientific(np.mean(flat_data), precision=1)
    std = np.format_float_scientific(np.std(flat_data), precision=1)

    # Generate Histogram
    xindx = int(sub_axs/dims)
    yindx = int(sub_axs%dims)
    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs[xindx][yindx].hist(flat_data, bins=100, density=True)

    axs[xindx][yindx].set_title(f"{title}: mu={mu}, std={std}")
    axs[xindx][yindx].ticklabel_format(axis="x", style="sci")
    xmin, xmax = (np.amin(flat_data), np.amax(flat_data))
    axs[xindx][yindx].set_xlim(xmin, xmax)

    # Color Histogram bins by height
    fracs = N / N.max()
    # we need to normalize the data to 0.1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    axs[xindx][yindx].yaxis.set_major_formatter(PercentFormatter(xmax=np.sum(N)))
