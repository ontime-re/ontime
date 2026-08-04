import unittest

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries as DartsTimeSeries

from ontime.core.metrics import nmae


class TestNmae(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-01", periods=4, freq="D")

    def test_nmae__known_values__should_return_expected_score(self):
        actual = DartsTimeSeries.from_times_and_values(
            self.index, np.array([1.0, 2.0, 3.0, 4.0])
        )
        pred = DartsTimeSeries.from_times_and_values(
            self.index, np.array([1.0, 2.0, 3.0, 5.0])
        )
        self.assertAlmostEqual(nmae(actual, pred), 0.1)

    def test_nmae__identical_series__should_return_zero(self):
        actual = DartsTimeSeries.from_times_and_values(
            self.index, np.array([1.0, 2.0, 3.0, 4.0])
        )
        self.assertAlmostEqual(nmae(actual, actual), 0.0)

    def test_nmae__actual_sums_to_zero__should_raise_value_error(self):
        zeros = DartsTimeSeries.from_times_and_values(
            self.index, np.array([0.0, 0.0, 0.0, 0.0])
        )
        pred = DartsTimeSeries.from_times_and_values(
            self.index, np.array([1.0, 2.0, 3.0, 4.0])
        )
        with pytest.raises(ValueError, match="sum to zero"):
            nmae(zeros, pred)
