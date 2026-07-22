import unittest

import numpy as np
import pandas as pd

from ontime.core.detection.registry.quantile import Quantile
from ontime.core.time_series import TimeSeries, BinaryTimeSeries


def make_ts(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return TimeSeries.from_times_and_values(index, np.array(values, dtype=float))


class TestQuantile(unittest.TestCase):
    def test_fit_detect__extreme_values__should_flag_outliers_at_both_ends(self):
        ts = make_ts([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        detector = Quantile(low_quantile=0.05, high_quantile=0.95)
        detector.fit(ts)
        detected = detector.detect(ts)
        self.assertIsInstance(detected, BinaryTimeSeries)
        np.testing.assert_array_equal(
            detected.values().flatten(),
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        )

    def test_fit_detect__narrow_quantile_bounds__should_flag_more_points(self):
        ts = make_ts([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        detector = Quantile(low_quantile=0.3, high_quantile=0.7)
        detector.fit(ts)
        detected = detector.detect(ts)
        self.assertGreater(detected.values().sum(), 2)
