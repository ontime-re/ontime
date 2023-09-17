from typing import Optional, Union

import numpy as np
import pandas as pd
from darts.utils.timeseries_generation import sine_timeseries

from ...abstract import AbstractBaseGenerator
from ...time_series import TimeSeries


class Sine(AbstractBaseGenerator):
    """
    Wrapper around Darts sine time series generator.
    """

    def __init__(self):
        super().__init__()

    def generate(
        self,
        value_frequency: float = 0.1,
        value_amplitude: float = 1.0,
        value_phase: float = 0.0,
        value_y_offset: float = 0.0,
        start: Optional[Union[pd.Timestamp, int]] = pd.Timestamp("2000-01-01"),
        end: Optional[Union[pd.Timestamp, int]] = None,
        length: Optional[int] = None,
        freq: Union[str, int] = None,
        column_name: Optional[str] = "sine",
        dtype: np.dtype = np.float64,
    ):
        """
        Generates a sine time series.

        :param value_frequency: float
        :param value_amplitude: float
        :param value_phase: float
        :param value_y_offset: float
        :param start: Pandas Timestamp or int
        :param end: Pandas Timestamp or int
        :param length: int
        :param freq: str or int
        :param column_name: str
        :param dtype: np.float64
        :return: TimeSeries
        """
        ts = sine_timeseries(
            value_frequency,
            value_amplitude,
            value_phase,
            value_y_offset,
            start,
            end,
            length,
            freq,
            column_name,
            dtype,
        )
        return TimeSeries.from_darts(ts)
