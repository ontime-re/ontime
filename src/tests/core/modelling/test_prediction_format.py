import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from torch import nn

from ontime.core.modelling import Model
from ontime.core.modelling.utils import check_prediction_format, normalize_prediction
from ontime.core.time_series import TimeSeries

N_POINTS = 40
HORIZON = 5


def make_ts(components, index_name="time"):
    index = pd.date_range("2024-01-01", periods=N_POINTS, freq="h", name=index_name)
    values = np.random.default_rng(42).normal(size=(N_POINTS, len(components)))
    return TimeSeries.from_times_and_values(
        times=index, values=values, columns=components
    )


def assert_consistent(pred, reference, n=HORIZON):
    assert len(pred) == n
    assert check_prediction_format(pred, reference)


class TestHelpers:
    def test_normalize_prediction_fixes_index_and_components(self):
        reference = make_ts(["temperature", "humidity"])
        wrong_index = pd.date_range("2030-06-01", periods=HORIZON, freq="D")
        pred = TimeSeries.from_times_and_values(
            times=wrong_index,
            values=np.zeros((HORIZON, 2)),
            columns=["0", "1"],
        )
        normalized = normalize_prediction(pred, reference)
        assert_consistent(normalized, reference)

    def test_normalize_prediction_component_mismatch_raises(self):
        reference = make_ts(["a", "b"])
        pred = make_ts(["a"])
        with pytest.raises(ValueError, match="component"):
            normalize_prediction(pred, reference)

    def test_check_prediction_format_detects_wrong_start(self):
        reference = make_ts(["a"])
        wrong_start = pd.date_range(
            reference.end_time() + 3 * reference.freq,
            periods=HORIZON,
            freq="h",
            name="time",
        )
        pred = TimeSeries.from_times_and_values(
            times=wrong_start, values=np.zeros((HORIZON, 1)), columns=["a"]
        )
        with pytest.raises(ValueError, match="start"):
            check_prediction_format(pred, reference)


class TestSkForecastUnivariate:
    def test_predict_without_ts(self):
        ts = make_ts(["temperature"])
        model = Model(LinearRegression(), lags=4).fit(ts)
        pred = model.predict(HORIZON)
        assert_consistent(pred, ts)

    def test_predict_with_ts(self):
        ts = make_ts(["temperature"])
        model = Model(LinearRegression(), lags=4).fit(ts)
        window = ts[-10:]
        pred = model.predict(HORIZON, window)
        assert_consistent(pred, window)


class TestSkForecastMultivariate:
    def test_predict_without_ts(self):
        ts = make_ts(["temperature", "humidity"])
        model = Model(LinearRegression(), lags=4).fit(ts)
        pred = model.predict(HORIZON)
        assert_consistent(pred, ts)

    def test_predict_with_ts(self):
        ts = make_ts(["temperature", "humidity"])
        model = Model(LinearRegression(), lags=4).fit(ts)
        window = ts[-10:]
        pred = model.predict(HORIZON, window)
        assert_consistent(pred, window)


class _TinyTorchModel(nn.Module):
    def __init__(self, input_len, output_len, n_features):
        super().__init__()
        self.output_len = output_len
        self.n_features = n_features
        self.linear = nn.Linear(input_len * n_features, output_len * n_features)

    def forward(self, x):
        out = self.linear(x.flatten(1))
        return out.reshape(x.shape[0], self.output_len, self.n_features)


class TestTorchWrapper:
    def test_predict(self):
        ts = make_ts(["temperature", "humidity"])
        model = Model(
            _TinyTorchModel(input_len=N_POINTS, output_len=HORIZON, n_features=2),
            input_chunk_length=N_POINTS,
            output_chunk_length=HORIZON,
            n_epochs=1,
        ).fit(ts)
        pred = model.predict(HORIZON, ts)
        assert_consistent(pred, ts)
