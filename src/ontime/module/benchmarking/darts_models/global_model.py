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
        training_epochs: int = 10,
        *args,
        **kwargs,
    ):
        self.name = name
        self.model = model
        self.mode = mode
        self.training_epochs = training_epochs

    def fit(
        self, train_ts: on.TimeSeries, val_ts: on.TimeSeries, *args, **kwargs
    ) -> None:
        """
        Fit a model on training data.
        """
        if isinstance(self.model, TorchForecastingModel):
            self.model.fit(series=train_ts, val_series=val_ts, epochs=self.training_epochs, *args, **kwargs)
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
            for metric in metrics:
                metrics_values[metric.name].append(metric.compute(forecast, label))
        return {metric: np.mean(values) for metric, values in metrics_values.items()}

    def load_checkpoint(self, path: str) -> GlobalDartsBenchmarkModel:
        """
        Load a model checkpoint from the given path.
        """
        return self.load(path)
    
    def reset_model(self):
        """
        Reset model weights, so that the model can be retrained without being recreated.
        """
        self.model = self.model.untrained_model()

    def get_benchmark_mode(self) -> BenchmarkMode:
        return self.mode
