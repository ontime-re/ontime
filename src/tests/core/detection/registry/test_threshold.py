import unittest

import numpy as np
import pandas as pd

from ontime.core.detection.registry.threshold import Threshold
from ontime.core.time_series import TimeSeries, BinaryTimeSeries


def make_ts(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return TimeSeries.from_times_and_values(index, np.array(values, dtype=float))


class TestThreshold(unittest.TestCase):
    def test_detect__values_within_bounds__should_return_only_zeros(self):
        ts = make_ts([1, 2, 3, 4, 5])
        detector = Threshold(low_threshold=0, high_threshold=10)
        detected = detector.detect(ts)
        self.assertIsInstance(detected, BinaryTimeSeries)
        np.testing.assert_array_equal(detected.values().flatten(), np.zeros(5))

    def test_detect__values_out_of_bounds__should_flag_them_as_anomalies(self):
        ts = make_ts([1, 20, 3, -5, 5])
        detector = Threshold(low_threshold=0, high_threshold=10)
        detected = detector.detect(ts)
        np.testing.assert_array_equal(
            detected.values().flatten(), np.array([0, 1, 0, 1, 0])
        )
