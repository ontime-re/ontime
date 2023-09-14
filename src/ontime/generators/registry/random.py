from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import random_walk_timeseries

from ...abstract import AbstractBaseGenerator


class Random(AbstractBaseGenerator):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
        end: Optional[Union[pd.Timestamp, int]] = None,
        length: Optional[int] = None,
        freq: Union[str, int] = None,
        column_name: Optional[str] = "random_walk",
        dtype: np.dtype = np.float64,
    ):
        return random_walk_timeseries(
            mean, std, start, end, length, freq, column_name, dtype
        )
