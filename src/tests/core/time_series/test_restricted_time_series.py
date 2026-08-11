import unittest

import numpy as np
import pandas as pd

from ontime.core.time_series.restricted_time_series import RestrictedTimeSeries
from ontime.core.utils.restriction import Restriction


def make_series_args(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", name="time")
    return {
        "times": index,
        "values": np.array(values, dtype=float).reshape(-1, 1),
        "components": ["a"],
    }


class TestRestrictedTimeSeries(unittest.TestCase):
    def test_constructor__no_restrictions__should_create_object_without_error(self):
        rts = RestrictedTimeSeries(**make_series_args([1.0, 2.0, 3.0]))
        self.assertEqual(rts.restrictions, [])

    def test_constructor__satisfied_restriction__should_create_object(self):
        restriction = Restriction("all positive", lambda values: np.all(values > 0))
        rts = RestrictedTimeSeries(
            **make_series_args([1.0, 2.0, 3.0]), restrictions=[restriction]
        )
        self.assertTrue(rts.check())

    def test_constructor__violated_restriction__should_raise_assertion_error(self):
        restriction = Restriction("all negative", lambda values: np.all(values < 0))
        with self.assertRaises(AssertionError):
            RestrictedTimeSeries(
                **make_series_args([1.0, 2.0, 3.0]), restrictions=[restriction]
            )

    def test_constructor__non_restriction_object__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            RestrictedTimeSeries(
                **make_series_args([1.0, 2.0, 3.0]), restrictions=["not a restriction"]
            )

    def test_add_restriction__violated_after_construction__should_raise_assertion_error(
        self,
    ):
        rts = RestrictedTimeSeries(**make_series_args([1.0, 2.0, 3.0]))
        restriction = Restriction("all negative", lambda values: np.all(values < 0))
        with self.assertRaises(AssertionError):
            rts.add_restriction(restriction)
