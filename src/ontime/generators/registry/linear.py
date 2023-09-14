from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import linear_timeseries

from ...abstract import AbstractBaseGenerator


class Linear(AbstractBaseGenerator):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        start_value: float = 0,
        end_value: float = 1,
        start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
        end: Optional[Union[pd.Timestamp, int]] = None,
        length: Optional[int] = None,
        freq: Union[str, int] = None,
        column_name: Optional[str] = "linear",
        dtype: np.dtype = np.float64,
    ):
        return linear_timeseries(
            start_value, end_value, start, end, length, freq, column_name, dtype
        )
