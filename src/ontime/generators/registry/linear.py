from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import linear_timeseries

from ...abstract import AbstractBaseGenerator
from ...time_series import TimeSeries


class Linear(AbstractBaseGenerator):
    """
    Wrapper around Darts linear time series generator.
    """

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
        """
        Generates a linear time series.

        :param start_value: float
        :param end_value: float
        :param start: Pandas Timestamp or int
        :param end: Pandas Timestamp or int
        :param length: int
        :param freq: str or int
        :param column_name: str
        :param dtype: np.float64
        :return: TimeSeries
        """
        ts = linear_timeseries(
            start_value, end_value, start, end, length, freq, column_name, dtype
        )
        return TimeSeries.from_darts(ts)
