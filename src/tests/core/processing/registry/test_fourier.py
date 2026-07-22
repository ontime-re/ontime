import unittest

import numpy as np
import pandas as pd

from ontime.core.processing.registry.fourier import Fourier
from ontime.core.time_series import TimeSeries


class TestFourier(unittest.TestCase):
    def setUp(self):
        self.n = 128
        index = pd.date_range("2024-01-01", periods=self.n, freq="h")
        # Sine wave with a period of 8 samples -> frequency 0.125 cycles/sample
        t = np.arange(self.n)
        values = np.sin(2 * np.pi * t / 8)
        self.ts = TimeSeries.from_times_and_values(index, values)

    def test_process__default_parameters__should_return_multivariate_series(self):
        result = Fourier(window_size=32, step_size=8, n_bins=4).process(self.ts)
        self.assertEqual(result.width, 4)
        expected_windows = (self.n - 32) // 8 + 1
        self.assertEqual(len(result), expected_windows)

    def test_process__sine_wave__should_concentrate_energy_in_matching_bin(self):
        result = Fourier(window_size=32, step_size=8, n_bins=4).process(self.ts)
        # Bins cover [0, 0.5]; 0.125 cycles/sample falls in the second bin
        mean_amplitudes = result.values().mean(axis=0)
        self.assertEqual(np.argmax(mean_amplitudes), 1)

    def test_process__result_timestamps__should_be_end_of_each_window(self):
        result = Fourier(window_size=32, step_size=8, n_bins=4).process(self.ts)
        self.assertEqual(result.time_index[0], self.ts.time_index[31])
        self.assertEqual(result.time_index[1], self.ts.time_index[39])

    def test_process__frequency_cap__should_restrict_frequency_range(self):
        result = Fourier(
            window_size=32, step_size=8, n_bins=2, frequency_cap=(0.1, 0.2)
        ).process(self.ts)
        self.assertEqual(result.width, 2)
        for name in result.components:
            self.assertTrue(name.startswith("freq_0.1"))

    def test_process__multivariate_input__should_raise_assertion_error(self):
        index = pd.date_range("2024-01-01", periods=64, freq="h")
        values = np.random.rand(64, 2)
        multivariate_ts = TimeSeries.from_times_and_values(index, values)
        with self.assertRaises(AssertionError):
            Fourier(window_size=32).process(multivariate_ts)

    def test_process__series_shorter_than_window__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Fourier(window_size=256).process(self.ts)

    def test_constructor__invalid_window_size__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Fourier(window_size=1)

    def test_constructor__invalid_step_size__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Fourier(window_size=32, step_size=0)

    def test_constructor__invalid_n_bins__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Fourier(window_size=32, n_bins=0)

    def test_constructor__invalid_frequency_cap__should_raise_assertion_error(self):
        with self.assertRaises(AssertionError):
            Fourier(window_size=32, frequency_cap=(0.3, 0.1))
