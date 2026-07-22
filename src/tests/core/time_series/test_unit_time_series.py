import unittest

import pandas as pd

from ontime.core.time_series import TimeSeries, UnitTimeSeries


def make_xa(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    df = pd.DataFrame({"a": values}, index=index)
    df.index.name = "time"
    return TimeSeries.from_pandas(df).data_array()


class TestUnitTimeSeries(unittest.TestCase):
    def test_constructor__values_between_zero_and_one__should_create_object(self):
        uts = UnitTimeSeries(make_xa([0.0, 0.5, 1.0]))
        self.assertIsInstance(uts, UnitTimeSeries)

    def test_constructor__value_above_one__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            UnitTimeSeries(make_xa([0.5, 1.5, 0.2]))

    def test_constructor__value_below_zero__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            UnitTimeSeries(make_xa([-0.1, 0.5, 0.2]))

    def test_unit_restriction__values_in_range__should_return_true(self):
        xa = make_xa([0.0, 0.5, 1.0])
        self.assertTrue(UnitTimeSeries.unit_restriction(xa))

    def test_unit_restriction__values_out_of_range__should_return_false(self):
        xa = make_xa([0.0, 2.0, 1.0])
        self.assertFalse(UnitTimeSeries.unit_restriction(xa))
