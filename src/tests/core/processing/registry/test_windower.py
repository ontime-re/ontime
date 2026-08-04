import unittest

import numpy as np
import pandas as pd

from ontime.core.processing.registry.windower import Windower
from ontime.core.time_series import TimeSeries


class TestWindower(unittest.TestCase):
    def test_process__rolling_mean__should_compute_expected_values(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        ts = TimeSeries.from_times_and_values(index, np.array([1.0, 2, 3, 4, 5]))
        windower = Windower(
            transforms={"function": "mean", "mode": "rolling", "window": 2}
        )
        windowed = windower.process(ts)
        values = windowed.values().flatten()
        np.testing.assert_array_almost_equal(
            values, np.array([1.0, 1.5, 2.5, 3.5, 4.5])
        )
