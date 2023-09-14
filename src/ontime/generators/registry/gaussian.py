from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import gaussian_timeseries

from ...abstract import AbstractBaseGenerator


class Gaussian(AbstractBaseGenerator):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0,
        start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
        end: Optional[Union[pd.Timestamp, int]] = None,
        length: Optional[int] = None,
        freq: Union[str, int] = None,
        column_name: Optional[str] = "gaussian",
        dtype: np.dtype = np.float64,
    ):
        return gaussian_timeseries(
            mean, std, start, end, length, freq, column_name, dtype
        )
