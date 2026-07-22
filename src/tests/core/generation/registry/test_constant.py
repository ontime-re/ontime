import unittest

import numpy as np
import pandas as pd

from ontime.core.generation.registry.constant import Constant
from ontime.core.time_series import TimeSeries

LENGTH = 10


class TestConstant(unittest.TestCase):
    def test_generate__default_params__should_return_time_series_of_requested_length(
        self,
    ):
        ts = Constant().generate(value=5, length=LENGTH)
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(len(ts), LENGTH)

    def test_generate__given_value__should_fill_all_entries_with_that_value(self):
        ts = Constant().generate(value=3.5, length=LENGTH)
        values = ts.values().flatten()
        np.testing.assert_array_equal(values, np.full(LENGTH, 3.5))

    def test_generate__explicit_start__should_use_it_as_first_timestamp(self):
        start = pd.Timestamp("2021-05-01")
        ts = Constant().generate(value=1, start=start, length=LENGTH)
        self.assertEqual(ts.start_time(), start)
