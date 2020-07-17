import os
from pickle import dump
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from lrgwd.split.config import GWFU_SCALER_FN, GWFV_SCALER_FN, TENSORS_SCALER_FN
from lrgwd.utils.io import from_pickle


class Scaler():
    def __init__(self, scaler: Dict[str, Union[str, bool]], save_path: Union[os.PathLike, str]):
        self.save_path = save_path
        if scaler["load"]:
            # load scalers
            self.tensors_scaler = from_pickle(os.path.join(scaler["path"], TENSORS_SCALER_FN))
            self.gwfu_scaler = from_pickle(os.path.join(scaler["path"], GWFU_SCALER_FN))
            self.gwfv_scaler = from_pickle(os.path.join(scaler["path"], GWFV_SCALER_FN))
        else: 
            self.tensors_scaler = StandardScaler()
            self.gwfu_scaler = StandardScaler()
            self.gwfv_scaler = StandardScaler()

    def partial_fit(
        self,
        tensors: pd.DataFrame,
        gwfu: pd.DataFrame,
        gwfv: pd.DataFrame,
    ) -> None:
        self.tensors_scaler.partial_fit(tensors)
        self.gwfu_scaler.partial_fit(gwfu)
        self.gwfv_scaler.partial_fit(gwfv)


    def save(self):
        dump(self.tensors_scaler, open(os.path.join(self.save_path, TENSORS_SCALER_FN), "wb"))
        dump(self.gwfu_scaler, open(os.path.join(self.save_path, GWFU_SCALER_FN), "wb"))
        dump(self.gwfv_scaler, open(os.path.join(self.save_path, GWFV_SCALER_FN), "wb"))


    def transform(
        self,
        tensors: pd.DataFrame,
        gwfu: pd.DataFrame,
        gwfv: pd.DataFrame,
    ):
        self.tensors_scaler.transform(tensors)
        self.gwfu_scaler.transform(gwfu)
        self.gwfv_scaler.transform(gwfv)
