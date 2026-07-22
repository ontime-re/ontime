import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

from ontime.core.modelling.libs.skforecast.skforecast_forecasting_model import (
    SkForecastForecastingModel,
)
from ontime.core.modelling.utils import check_prediction_format
from ontime.core.time_series import TimeSeries

N_POINTS = 40
HORIZON = 5


def make_ts(components):
    index = pd.date_range("2024-01-01", periods=N_POINTS, freq="h", name="time")
    values = np.random.default_rng(42).normal(size=(N_POINTS, len(components)))
    return TimeSeries.from_times_and_values(
        times=index, values=values, columns=components
    )


class TestConstructor:
    def test_constructor__creation_from_instance__should_store_estimator(self):
        model = SkForecastForecastingModel(LinearRegression(), lags=4)
        assert isinstance(model.sk_model, LinearRegression)
        assert model.model is None

    def test_constructor__creation_from_class__should_instantiate_estimator(self):
        model = SkForecastForecastingModel(LinearRegression, lags=4)
        assert isinstance(model.sk_model, LinearRegression)


class TestFit:
    def test_fit__univariate_series__should_use_recursive_forecaster(self):
        model = SkForecastForecastingModel(LinearRegression(), lags=4)
        model.fit(make_ts(["temperature"]))
        assert isinstance(model.model, ForecasterRecursive)
        assert model.is_multivariate is False

    def test_fit__multivariate_series__should_use_multiseries_forecaster(self):
        model = SkForecastForecastingModel(LinearRegression(), lags=4)
        model.fit(make_ts(["temperature", "humidity"]))
        assert isinstance(model.model, ForecasterRecursiveMultiSeries)
        assert model.is_multivariate is True


class TestPredict:
    def test_predict__before_fit__should_raise_value_error(self):
        model = SkForecastForecastingModel(LinearRegression(), lags=4)
        with pytest.raises(ValueError, match="fitted"):
            model.predict(HORIZON)

    def test_predict__univariate_without_ts__should_return_consistent_prediction(self):
        ts = make_ts(["temperature"])
        model = SkForecastForecastingModel(LinearRegression(), lags=4).fit(ts)
        pred = model.predict(HORIZON)
        assert len(pred) == HORIZON
        assert check_prediction_format(pred, ts)

    def test_predict__univariate_with_ts__should_return_consistent_prediction(self):
        ts = make_ts(["temperature"])
        model = SkForecastForecastingModel(LinearRegression(), lags=4).fit(ts)
        window = ts[-10:]
        pred = model.predict(HORIZON, window)
        assert len(pred) == HORIZON
        assert check_prediction_format(pred, window)

    def test_predict__multivariate_without_ts__should_return_consistent_prediction(
        self,
    ):
        ts = make_ts(["temperature", "humidity"])
        model = SkForecastForecastingModel(LinearRegression(), lags=4).fit(ts)
        pred = model.predict(HORIZON)
        assert len(pred) == HORIZON
        assert pred.n_components == 2
        assert check_prediction_format(pred, ts)

    def test_predict__multivariate_with_ts__should_return_consistent_prediction(self):
        ts = make_ts(["temperature", "humidity"])
        model = SkForecastForecastingModel(LinearRegression(), lags=4).fit(ts)
        window = ts[-10:]
        pred = model.predict(HORIZON, window)
        assert len(pred) == HORIZON
        assert pred.n_components == 2
        assert check_prediction_format(pred, window)

    def test_predict__with_list_of_ts__should_raise_value_error(self):
        ts = make_ts(["temperature"])
        model = SkForecastForecastingModel(LinearRegression(), lags=4).fit(ts)
        with pytest.raises(ValueError, match="single TimeSeries"):
            model.predict(HORIZON, [ts, ts])
