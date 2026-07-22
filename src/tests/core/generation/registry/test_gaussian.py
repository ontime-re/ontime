import unittest

from ontime.core.generation.registry.gaussian import Gaussian
from ontime.core.time_series import TimeSeries

LENGTH = 200


class TestGaussian(unittest.TestCase):
    def test_generate__default_params__should_return_time_series_of_requested_length(
        self,
    ):
        ts = Gaussian().generate(mean=0.0, std=1.0, length=LENGTH)
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(len(ts), LENGTH)

    def test_generate__given_mean_and_std__should_approximate_them_over_large_sample(
        self,
    ):
        ts = Gaussian().generate(mean=10.0, std=2.0, length=5000)
        values = ts.values().flatten()
        self.assertAlmostEqual(values.mean(), 10.0, delta=0.5)
        self.assertAlmostEqual(values.std(), 2.0, delta=0.5)
