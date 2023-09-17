from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import random_walk_timeseries

from ...time_series import TimeSeries
from ...abstract import AbstractBaseGenerator


class RandomWalk(AbstractBaseGenerator):
    """
    Wrapper around Darts random walk time series generator.
    """

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
        """
        Generates a random walk time series.

        :param mean: float
        :param std: float
        :param start: Pandas Timestamp or int
        :param end: Pandas Timestamp or int
        :param length: int
        :param freq: str or int
        :param column_name: str
        :param dtype: np.float64
        :return: TimeSeries
        """
        ts = random_walk_timeseries(
            mean, std, start, end, length, freq, column_name, dtype
        )
        return TimeSeries.from_darts(ts)
