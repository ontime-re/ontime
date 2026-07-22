import unittest

from ontime.core.generation.registry.linear import Linear
from ontime.core.time_series import TimeSeries

LENGTH = 10


class TestLinear(unittest.TestCase):
    def test_generate__default_params__should_return_time_series_of_requested_length(
        self,
    ):
        ts = Linear().generate(start_value=0, end_value=9, length=LENGTH)
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(len(ts), LENGTH)

    def test_generate__start_and_end_values__should_match_first_and_last_entries(self):
        ts = Linear().generate(start_value=0, end_value=9, length=LENGTH)
        values = ts.values().flatten()
        self.assertAlmostEqual(values[0], 0)
        self.assertAlmostEqual(values[-1], 9)

    def test_generate__values__should_be_monotonically_increasing(self):
        ts = Linear().generate(start_value=0, end_value=9, length=LENGTH)
        values = ts.values().flatten()
        self.assertTrue(all(values[i] < values[i + 1] for i in range(len(values) - 1)))
