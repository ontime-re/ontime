import unittest

import numpy as np
import pandas as pd

from ontime.core.processing.registry.correlation import Correlation
from ontime.core.time_series import TimeSeries


class TestCorrelation(unittest.TestCase):
    def test_process__perfectly_correlated_columns__should_return_correlation_of_one(
        self,
    ):
        index = pd.date_range("2024-01-01", periods=6, freq="D")
        df = pd.DataFrame(
            {"a": [1.0, 2, 3, 4, 5, 6], "b": [2.0, 4, 6, 8, 10, 12]}, index=index
        )
        df.index.name = "time"
        ts = TimeSeries.from_dataframe(df)

        correlated = Correlation(window=3).process(ts)

        self.assertIsInstance(correlated, TimeSeries)
        self.assertIn("a_b", correlated.components)
        values = correlated["a_b"].values().flatten()
        # first two entries are NaN as the rolling window is not yet full
        np.testing.assert_array_almost_equal(values[2:], np.ones(4))
