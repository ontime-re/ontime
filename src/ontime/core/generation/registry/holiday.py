from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import holidays_timeseries

from ..abstract_generator import AbstractGenerator
from ...time_series import TimeSeries


class Holiday(AbstractGenerator):
    """
    Wrapper around Darts holiday time series generator.
    """

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
        """
        Generates a holiday time series.

        :param time_index: Pandas DatetimeIndex
        :param country_code: str
        :param prov: str
        :param state: str
        :param column_name: str
        :param until: int, str or Pandas Timestamp
        :param add_length: int
        :param dtype: np.float64
        :return: TimeSeries
        """
        ts = holidays_timeseries(
            time_index, country_code, prov, state, column_name, until, add_length, dtype
        )
        return TimeSeries(ts.data_array())
