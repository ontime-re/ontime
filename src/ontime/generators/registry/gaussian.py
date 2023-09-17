from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import gaussian_timeseries

from ...abstract import AbstractBaseGenerator
from ...time_series import TimeSeries


class Gaussian(AbstractBaseGenerator):
    """
    Wrapper around Darts gaussian time series generator.
    """

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
        """
        Generates a gaussian time series.

        :param mean: float or np.ndarray
        :param std: float or np.ndarray
        :param start: Pandas Timestamp or int
        :param end: Pandas Timestamp or int
        :param length: int
        :param freq: str or int
        :param column_name: str
        :param dtype: np.float64
        :return: TimeSeries
        """
        ts = gaussian_timeseries(
            mean, std, start, end, length, freq, column_name, dtype
        )
        return TimeSeries.from_darts(ts)
