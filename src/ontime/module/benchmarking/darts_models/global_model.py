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

from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

class GlobalDartsBenchmarkModel(AbstractBenchmarkModel):
    """
    A wrapper around GlobalForecastingModel from Darts, to be forecasted like any other model.
    The major specificity of GlobalForecastingModel models is that they can be trained on multiple target (i.e. multivariate) series. 
    For more information about them, see https://unit8co.github.io/darts/_modules/darts/models/forecasting/forecasting_model.html
    """

    def __init__(
        self,
        name: str,
        model: GlobalForecastingModel,
        mode: BenchmarkMode,
        *args,
        **kwargs,
    ):
        self.name = name
        self.model = model
        self.mode = mode

    def fit(
        self, train_ts: on.TimeSeries, val_ts: on.TimeSeries, *args, **kwargs
    ) -> None:
        """
        Fit a model on training data.
        """
        if isinstance(self.model, TorchForecastingModel):
            self.model.fit(series=train_ts, val_series=val_ts, *args, **kwargs)
        else:
            self.model.fit(series=train_ts, *args, **kwargs)

    def predict(
        self, ts: on.TimeSeries, horizon: int, *args, **kwargs
    ) -> on.TimeSeries:
        """
        Forecast the given time series.
        """
        return self.model.predict(series=ts, n=horizon, *args, **kwargs)

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
        metrics_values = {metric.name: [] for metric in metrics}
        for input, label in zip(dataset["input"], dataset["label"]):
            forecast = self.predict(input, horizon)
            computed_metrics = self._compute_metric(forecast, label, metrics, input=input)
            {metrics_values[metric].append(value) for metric, value in computed_metrics.items()}
        return {metric: np.nanmean(values) for metric, values in metrics_values.items()}

    def load_checkpoint(self, path: str) -> GlobalDartsBenchmarkModel:
        """
        Load a model checkpoint from the given path.
        """
        return self.load(path)
    
    def reset_model(self, **new_model_params):
        """
        Reset model weights, so that the model can be retrained without being recreated.
        """
        model_params = self.model._model_params.copy()
        model_params.update(new_model_params)
        self.model = self.model.__class__(**model_params)

    def get_benchmark_mode(self) -> BenchmarkMode:
        return self.mode
