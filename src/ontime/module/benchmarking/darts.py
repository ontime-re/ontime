from __future__ import annotations
from abc import ABCMeta
from typing import List, Union, Tuple, Generator

import numpy as np
import ontime as on
from ontime.module.benchmarking.benchmark import (
    AbstractBenchmarkModel,
    BenchmarkMetric,
    BenchmarkMode,
)
from ontime.module.processing.common import (
    split_in_windows,
    split_inputs_from_targets,
)
from darts.models.forecasting.forecasting_model import LocalForecastingModel


def create_dataset(
    ts: on.TimeSeries,
    stride_length: int,
    context_length: int,
    prediction_length: int,
    gap: int = 0,
) -> dict[str, List[on.TimeSeries]]:
    """
    Create a dataset of ontime TimeSeries in an expanding window style from a given time series. The dataset is a dictionary with two keys: "input" and "label".
    """
    # TODO: This method should be improved as we can have memory issues with large time series (e.g. we should process the time series per batch, using a generator).
    dataset = {"input": [], "label": []}
    ts_list = split_in_windows(ts, context_length+prediction_length+gap, stride_length)
    dataset["input"], dataset["label"] = split_inputs_from_targets(
        ts_list,
        input_length=context_length,
        target_length=prediction_length,
        gap_length=gap,
    )
    
    return dataset

class SimpleDartsBenchmarkModel(AbstractBenchmarkModel):
    """
    A wrapper around LocalForecastingModel from Darts, to be forecasted like any other model.
    The major specificity of LocalForecastingModel models is that they are fitted on a time series and then used to forecast this same time series.
    For more information about them, see https://unit8co.github.io/darts/_modules/darts/models/forecasting/forecasting_model.html
    """

    def __init__(
        self,
        name: str,
        model: LocalForecastingModel,
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
        self.model.fit(train_ts, val_ts, *args, **kwargs)

    def predict(
        self, ts: on.TimeSeries, horizon: int, *args, **kwargs
    ) -> on.TimeSeries:
        """
        Forecast the given time series.
        """
        if ts is None:
            # we use the predict method of the model directly, to make predictions on the train set
            return self.model.predict(horizon, *args, **kwargs)
        # TODO: create dataset in rolling window or expanding window style and predict for each window
        else:
            self.model.fit(ts)
            return self.model.predict(horizon, *args, **kwargs)

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
            self.model.fit(input)
            forecast = self.model.predict(horizon)
            for metric in metrics:
                metrics_values[metric.name].append(metric.compute(forecast, label))
        return {metric: np.mean(values) for metric, values in metrics_values.items()}

    def load_checkpoint(self, path: str) -> SimpleDartsBenchmarkModel:
        """
        Load a model checkpoint from the given path.
        """
        return self.load(path)

    def get_benchmark_mode(self) -> BenchmarkMode:
        return self.mode
