from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import constant_timeseries

from ...abstract import AbstractBaseGenerator


class Constant(AbstractBaseGenerator):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        value: float = 1,
        start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
        end: Optional[Union[pd.Timestamp, int]] = None,
        length: Optional[int] = None,
        freq: Union[str, int] = None,
        column_name: Optional[str] = "constant",
        dtype: np.dtype = np.float64,
    ):
        return constant_timeseries(value, start, end, length, freq, column_name, dtype)
