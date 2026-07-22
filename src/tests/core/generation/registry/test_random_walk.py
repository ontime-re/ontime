import unittest

from ontime.core.generation.registry.random_walk import RandomWalk
from ontime.core.time_series import TimeSeries

LENGTH = 50


class TestRandomWalk(unittest.TestCase):
    def test_generate__default_params__should_return_time_series_of_requested_length(
        self,
    ):
        ts = RandomWalk().generate(mean=0.0, std=1.0, length=LENGTH)
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(len(ts), LENGTH)

    def test_generate__two_calls__should_be_random_and_not_produce_identical_series(
        self,
    ):
        ts1 = RandomWalk().generate(mean=0.0, std=1.0, length=LENGTH)
        ts2 = RandomWalk().generate(mean=0.0, std=1.0, length=LENGTH)
        self.assertFalse((ts1.values() == ts2.values()).all())
