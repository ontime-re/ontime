import unittest

import numpy as np

from ontime.core.generation.registry.sine import Sine
from ontime.core.time_series import TimeSeries

LENGTH = 20


class TestSine(unittest.TestCase):
    def test_generate__default_params__should_return_time_series_of_requested_length(
        self,
    ):
        ts = Sine().generate(length=LENGTH)
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(len(ts), LENGTH)

    def test_generate__amplitude_and_offset__should_bound_values_accordingly(self):
        amplitude = 2.0
        offset = 1.0
        ts = Sine().generate(
            value_amplitude=amplitude, value_y_offset=offset, length=LENGTH
        )
        values = ts.values().flatten()
        self.assertTrue(np.all(values <= amplitude + offset + 1e-9))
        self.assertTrue(np.all(values >= -amplitude + offset - 1e-9))
