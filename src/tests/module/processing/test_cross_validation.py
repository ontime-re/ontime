import unittest

import numpy as np
import pandas as pd

from ontime.core.time_series import TimeSeries
from ontime.module.benchmarking.benchmark_metric import BenchmarkMetric
from ontime.module.processing.cross_validation import (
    cross_validation,
    evaluate_cross_validation,
)
from darts.metrics import mae


class _NaiveLastValueModel:
    """
    Minimal model used to exercise `evaluate_cross_validation` without depending on
    heavier libraries. Predicts by repeating the last value of the fitted TimeSeries.
    """

    def __init__(self):
        self._last_value = None

    def fit(self, ts):
        self._last_value = ts.pd_dataframe().iloc[-1].to_numpy()
        return self

    def predict(self, n, ts=None):
        df = (ts if ts is not None else None).pd_dataframe()
        freq = df.index.freq or pd.infer_freq(df.index)
        index = pd.date_range(
            start=df.index[-1] + (df.index[1] - df.index[0]), periods=n, freq=freq
        )
        values = np.tile(self._last_value, (n, 1))
        pred_df = pd.DataFrame(values, index=index, columns=df.columns)
        return TimeSeries.from_dataframe(pred_df)


def _make_ts(length=40):
    index = pd.date_range(start="2020-01-01", periods=length, freq="D")
    df = pd.DataFrame({"v": np.arange(length, dtype=float)}, index=index)
    return TimeSeries.from_dataframe(df)


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.ts = _make_ts(40)

    def test_crossValidation__unknownStrategy__shouldRaiseValueError(self):
        with self.assertRaises(ValueError):
            cross_validation(self.ts, n_splits=3, strategy="unknown")

    def test_crossValidation__nSplitsTooLarge__shouldRaiseValueError(self):
        with self.assertRaises(ValueError):
            cross_validation(self.ts, n_splits=100, strategy="expanding", horizon=1)

    def test_crossValidation__horizonExceedingRemainingData__shouldRaiseValueError(
        self,
    ):
        with self.assertRaises(ValueError):
            cross_validation(
                self.ts,
                n_splits=2,
                strategy="expanding",
                initial_train_size=35,
                horizon=10,
            )

    def test_crossValidation__expandingStrategy__trainSetShouldGrowEachFold(self):
        folds = cross_validation(
            self.ts,
            n_splits=4,
            strategy="expanding",
            initial_train_size=20,
            horizon=1,
        )
        self.assertEqual(len(folds), 4)
        train_lengths = [len(train) for train, _ in folds]
        self.assertEqual(train_lengths, [20, 21, 22, 23])
        for _, test in folds:
            self.assertEqual(len(test), 1)

    def test_crossValidation__expandingWithHorizonOne__isWalkForwardValidation(self):
        folds = cross_validation(
            self.ts,
            n_splits=3,
            strategy="expanding",
            initial_train_size=30,
            horizon=1,
        )
        for train, test in folds:
            self.assertEqual(len(test), 1)
        # each successive train set includes exactly one more point than the previous
        for i in range(1, len(folds)):
            self.assertEqual(len(folds[i][0]), len(folds[i - 1][0]) + 1)

    def test_crossValidation__expandingWithHorizonGreaterThanOne__isMultiHorizon(self):
        folds = cross_validation(
            self.ts,
            n_splits=3,
            strategy="expanding",
            initial_train_size=25,
            horizon=5,
        )
        for _, test in folds:
            self.assertEqual(len(test), 5)

    def test_crossValidation__slidingStrategy__trainSetSizeShouldStayConstant(self):
        folds = cross_validation(
            self.ts,
            n_splits=4,
            strategy="sliding",
            initial_train_size=10,
            horizon=2,
        )
        self.assertEqual(len(folds), 4)
        for train, test in folds:
            self.assertEqual(len(train), 10)
            self.assertEqual(len(test), 2)
        # train windows should shift forward between folds
        starts = [train.pd_dataframe().index[0] for train, _ in folds]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(len(set(starts)) == len(starts))

    def test_crossValidation__blockedStrategy__blocksShouldBeContiguousAndNonOverlapping(
        self,
    ):
        folds = cross_validation(self.ts, n_splits=3, strategy="blocked")
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            train_end = train.pd_dataframe().index[-1]
            test_start = test.pd_dataframe().index[0]
            self.assertLess(train_end, test_start)

    def test_crossValidation__gapParameter__shouldLeaveGapBetweenTrainAndTest(self):
        gap = 3
        folds = cross_validation(
            self.ts,
            n_splits=2,
            strategy="expanding",
            initial_train_size=20,
            horizon=1,
            gap=gap,
        )
        for train, test in folds:
            train_end = train.pd_dataframe().index[-1]
            test_start = test.pd_dataframe().index[0]
            gap_points = (test_start - train_end).days - 1
            self.assertEqual(gap_points, gap)

    def test_crossValidation__blockedWithGap__shouldRaiseIfGapExceedsBlockSize(self):
        with self.assertRaises(ValueError):
            cross_validation(self.ts, n_splits=3, strategy="blocked", gap=100)

    def test_evaluateCrossValidation__naiveModelAndMaeMetric__shouldReturnPerFoldAndAggregatedResults(
        self,
    ):
        metric = BenchmarkMetric(name="mae", metric_function=mae)
        results = evaluate_cross_validation(
            _NaiveLastValueModel(),
            self.ts,
            metrics=metric,
            n_splits=3,
            strategy="expanding",
            initial_train_size=20,
            horizon=2,
        )
        self.assertIn("folds", results)
        self.assertIn("mean", results)
        self.assertIn("std", results)
        self.assertEqual(len(results["folds"]["mae"]), 3)
        self.assertIn("mae", results["mean"])
        self.assertIn("mae", results["std"])
        # naive last-value forecasting on a strictly increasing series has a known, constant error
        for value in results["folds"]["mae"]:
            self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
