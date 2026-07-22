import unittest

import numpy as np
import pandas as pd

from ontime.core.processing.registry.density import Density
from ontime.core.time_series import BinaryTimeSeries


class TestDensity(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-01-01", periods=6, freq="D")
        values = np.array([0, 1, 1, 0, 1, 0], dtype=float)
        self.ts = BinaryTimeSeries.from_times_and_values(index, values)

    def test_process__absolute_mode__should_count_anomalies_in_window(self):
        density = Density(window_length=3, mode="absolute").process(self.ts)
        values = density.values().flatten()
        np.testing.assert_array_almost_equal(values, np.array([0, 1, 2, 2, 2, 1]))

    def test_process__relative_mode__should_return_ratio_of_anomalies_in_window(self):
        density = Density(window_length=3, mode="relative").process(self.ts)
        values = density.values().flatten()
        np.testing.assert_array_almost_equal(values, np.array([0, 1, 2, 2, 2, 1]) / 3)

    def test_constructor__invalid_window_length__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Density(window_length="3", mode="absolute")

    def test_constructor__invalid_mode__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Density(window_length=3, mode="unknown")
