from typing import Union
import numpy as np

from ...time_series import TimeSeries, UnitTimeSeries, BinaryTimeSeries
from ..abstract_processor import AbstractProcessor


class Density(AbstractProcessor):
    """Density class handles density computation in a TimeSeries"""

    def __init__(self, window_length: int, mode: str = "absolute"):
        """Constructor of a density processor

        Two modes are available:
        - 'absolute': the density is the absolute number of anomalies in the window
        - 'relative': the density is the relative number of anomalies in the window between 0 and 1

        :param window_length: int
        :param mode: str
        """
        assert isinstance(
            window_length, int
        ), f"window_length must be an integer, not {type(window_length)}"
        assert mode in {
            "absolute",
            "relative",
        }, f"mode has an invalid value: {mode}. Must be 'absolute' or 'relative'."

        self.window_length = window_length
        self.mode = mode

    def process(
        self, ts: Union[UnitTimeSeries, BinaryTimeSeries]
    ) -> Union[UnitTimeSeries, BinaryTimeSeries]:
        """Compute densities for a TimeSeries

        :param ts: TimeSeries
        :return: TimeSeries
        """

        match self.mode:
            case "absolute":

                def count(x):
                    return np.sum(x)

            case "relative":

                def count(x):
                    return np.sum(x) / self.window_length

        density_ts = TimeSeries.from_darts(
            ts.window_transform(
                transforms={
                    "function": lambda x: count(x),
                    "mode": "rolling",
                    "window": self.window_length,
                    "function_name": f"count_{self.mode}",
                }
            )
        )

        return density_ts
