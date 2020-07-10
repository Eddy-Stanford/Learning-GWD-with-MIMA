import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lrgwd.utils.io import from_pickle

gwfu_scaler = from_pickle("../runs/massive/split/gwfu_scaler.pkl")

for gwfu_chunk, tensors_chunk in zip(
    pd.read_csv("../runs/massive/split/train_gwfu.csv", chunksize=100000),
    pd.read_csv("../runs/massive/split/train_tensors.csv", chunksize=100000)
):
    gwfu_chunk = gwfu_chunk.to_numpy()
    tensors_chunk = tensors_chunk.to_numpy()
    break

scaled_gwfu_chunk = gwfu_scaler.transform(gwfu_chunk)
plevels = gwfu_chunk[0].shape[0]

for plevel in reversed(range(plevels)):

    # raw_gwfu = gwfu_chunk[:,plevel]
    scaled_gwfu = scaled_gwfu_chunk[:,plevel]

    fig = plt.figure(figsize=(8,6))
    plt.hist(
        [scaled_gwfu],
        bins=1000,
        label=["scaled_gwfu"]
    )
    plt.xlabel("gwfu (m/s^2)", size=14)
    plt.ylabel("Count", size=14)
    plt.title(f"Histogram scaled_gwfu for Plevel {plevel}")
    plt.legend(loc='upper right')

    # Make figures full screen
    fig.set_size_inches(32, 18)
    os.makedirs("experiments", exist_ok=True)
    fig.savefig(os.path.join(f"experiments/scaled_{plevel}.png"))
    plt.close(fig)
