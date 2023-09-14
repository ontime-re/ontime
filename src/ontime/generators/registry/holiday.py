from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import holidays_timeseries

from ...abstract import AbstractBaseGenerator


class Holiday(AbstractBaseGenerator):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        time_index: pd.DatetimeIndex,
        country_code: str,
        prov: str = None,
        state: str = None,
        column_name: Optional[str] = "holidays",
        until: Optional[Union[int, str, pd.Timestamp]] = None,
        add_length: int = 0,
        dtype: np.dtype = np.float64,
    ):
        return holidays_timeseries(
            time_index, country_code, prov, state, column_name, until, add_length, dtype
        )
