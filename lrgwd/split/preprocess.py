import os
from pickle import dump
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor():
    def __init__(self, save_path: Union[os.PathLike, str]):
        self.save_path = save_path
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
        dump(self.tensors_scaler, open(os.path.join(self.save_path, "tensors_scaler.pkl"), "wb"))
        dump(self.gwfu_scaler, open(os.path.join(self.save_path, "gwfu_scaler.pkl"), "wb"))
        dump(self.gwfv_scaler, open(os.path.join(self.save_path, "gwfv_scaler.pkl"), "wb"))


    def transform(
        self,
        tensors: pd.DataFrame,
        gwfu: pd.DataFrame,
        gwfv: pd.DataFrame,
    ):
        self.tensors_scaler.transform(tensors)
        self.gwfu_scaler.transform(gwfu)
        self.gwfv_scaler.transform(gwfv)
