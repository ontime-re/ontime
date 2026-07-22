import unittest

import numpy as np
import pandas as pd

from ontime.core.processing.registry.filler import Filler
from ontime.core.time_series import TimeSeries


class TestFiller(unittest.TestCase):
    def test_process__missing_values__should_interpolate_them(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        ts = TimeSeries.from_times_and_values(index, values)
        filled = Filler().process(ts)
        self.assertIsInstance(filled, TimeSeries)
        np.testing.assert_array_equal(
            filled.values().flatten(), np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )

    def test_process__no_missing_values__should_leave_series_unchanged(self):
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        values = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries.from_times_and_values(index, values)
        filled = Filler().process(ts)
        np.testing.assert_array_equal(filled.values().flatten(), values)
