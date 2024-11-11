from __future__ import annotations
from typing import List
import numpy as np

import ontime as on
from ontime.module.benchmarking.benchmark import (
    AbstractBenchmarkModel,
    BenchmarkMetric,
    BenchmarkMode,
)
from .common import create_dataset

from darts.models.forecasting.forecasting_model import LocalForecastingModel

class LocalDartsBenchmarkModel(AbstractBenchmarkModel):
    """
    A wrapper around LocalForecastingModel from Darts, to be forecasted like any other model.
    The major specificity of LocalForecastingModel models is that 1. they are fitted on a time series and then used to forecast this same time series, 2. they can be trained
    on a single target (i.e. univariate) series only. 
    For more information about them, see https://unit8co.github.io/darts/_modules/darts/models/forecasting/forecasting_model.html
    """

    def __init__(
        self,
        name: str,
        model: LocalForecastingModel,
        mode: BenchmarkMode,
        multivariate: bool = False,
        *args,
        **kwargs,
    ):
        self.name = name
        self.model = model
        self.mode = mode
        self.multivariate = multivariate

    def fit(
        self, train_ts: on.TimeSeries, val_ts: on.TimeSeries, *args, **kwargs
    ) -> None:
        """
        Fit a model on training data.
        """
        if train_ts.n_components > 1:
            raise ValueError("This model can only be fitted on univariate time series. Use directly predict() method to make a prediction on multivariate time series, by instantiating a model with argument multivariate=True.")
        self.model.fit(train_ts, *args, **kwargs)

    def predict(
        self, ts: on.TimeSeries, horizon: int, *args, **kwargs
    ) -> on.TimeSeries:
        """
        Forecast the given time series.
        """
        if ts is None:
            # we use the predict method of the model directly, to make predictions on the train set
            return self.model.predict(horizon, *args, **kwargs)
        return self._fit_predict(ts, horizon, *args, **kwargs)

    def evaluate(
        self,
        ts: on.TimeSeries,
        horizon: int,
        metrics: List[BenchmarkMetric],
        *args,
        **kwargs,
    ) -> dict:
        """
        Evaluate the model on test data, using the given metrics.
        """
        dataset = create_dataset(ts, prediction_length=horizon, *args, **kwargs)
        dataset["forecast"] = []
        for input in dataset["input"]:
            dataset["forecast"].append(self._fit_predict(input, horizon))
        return self._compute_metrics(dataset["forecast"], dataset["label"], metrics, input=dataset["input"])
    
    def _fit_predict(
        self,
        ts: on.TimeSeries,
        horizon: int,
        *args,
        **kwargs
    ) -> on.TimeSeries:
        """
        Fit the model on the given time series and produce from it a forecast of a given horizon
        """
        if ts.n_components > 1:
            if self.multivariate:
                predictions = []
                # loop over each component
                for i in range(ts.n_components):
                    component = ts.univariate_component(i)
                    self.model.fit(component)
                    forecast = self.model.predict(horizon, *args, **kwargs)
                    predictions.append(forecast)
                # combine predictions
                combined_forecast = on.TimeSeries.from_times_and_values(
                    times=predictions[0].time_index,
                    values=np.column_stack([f.values() for f in predictions]),
                )
                return combined_forecast.with_columns_renamed(combined_forecast.components, ts.components)
            else:
                raise ValueError("This model can only make prediction on univariate time series. To perform a prediction on multivariate time series, you should instantiate a model with argument multivariate=True.")
        else:
            self.model.fit(ts)
            return self.model.predict(horizon, *args, **kwargs)
        

    def load_checkpoint(self, path: str) -> LocalDartsBenchmarkModel:
        """
        Load a model checkpoint from the given path.
        """
        return self.load(path)

    def get_benchmark_mode(self) -> BenchmarkMode:
        return self.mode
