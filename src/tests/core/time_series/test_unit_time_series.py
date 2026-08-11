import unittest

import numpy as np
import pandas as pd

from ontime.core.time_series import UnitTimeSeries


def make_series_args(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", name="time")
    return {
        "times": index,
        "values": np.array(values, dtype=float).reshape(-1, 1),
        "components": ["a"],
    }


class TestUnitTimeSeries(unittest.TestCase):
    def test_constructor__values_between_zero_and_one__should_create_object(self):
        uts = UnitTimeSeries(**make_series_args([0.0, 0.5, 1.0]))
        self.assertIsInstance(uts, UnitTimeSeries)

    def test_constructor__value_above_one__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            UnitTimeSeries(**make_series_args([0.5, 1.5, 0.2]))

    def test_constructor__value_below_zero__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            UnitTimeSeries(**make_series_args([-0.1, 0.5, 0.2]))

    def test_unit_restriction__values_in_range__should_return_true(self):
        values = make_series_args([0.0, 0.5, 1.0])["values"]
        self.assertTrue(UnitTimeSeries.unit_restriction(values))

    def test_unit_restriction__values_out_of_range__should_return_false(self):
        values = make_series_args([0.0, 2.0, 1.0])["values"]
        self.assertFalse(UnitTimeSeries.unit_restriction(values))
