from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Union, Dict
from enum import Enum
import numpy as np
import logging

from numpy import ndarray
from darts.metrics import mase

from ontime.core.time_series.time_series import TimeSeries
from ontime.module.benchmarking import BenchmarkMetric


class BenchmarkMode(Enum):
    ZERO_SHOT = 1  # no training, only inference
    FULL_SHOT = 3  # full training


class AbstractBenchmarkModel(ABC):  
    @abstractmethod
    def fit(self, train_ts: TimeSeries, val_ts: TimeSeries, *args, **kwargs) -> None:
        """
        Fit a model on training data.
        """
        pass

    @abstractmethod
    def predict(self, ts: TimeSeries, horizon: int, *args, **kwargs) -> TimeSeries:
        """
        Predict the next `horizon` steps of the time series.
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        ts: TimeSeries,
        horizon: int,
        metrics: List[BenchmarkMetric],
        *args,
        **kwargs,
    ) -> dict:
        """
        Evaluate the model on test data, using the given metrics.
        """
        pass
    
    def reset_model(self, **new_model_params):
        """
        Reset model weights, so that the model can be retrained without being recreated.
        """
        pass
        

    @abstractmethod
    def load_checkpoint(self, path: str) -> AbstractBenchmarkModel:
        """
        Load a model checkpoint from the given path, and return the model.
        """
        pass

    @abstractmethod
    def get_benchmark_mode(self) -> BenchmarkMode:
        """
        Return the benchmark mode of the model.
        """
        pass
    
    def _compute_metrics(self, forecasts: List[TimeSeries], labels: List[TimeSeries], metrics: List[BenchmarkMetric], **kwargs) -> Dict[str, ndarray]:
        """
        Helper method to compute metrics on given forecast and label TimeSeries. This method also handles any specific
        conditions required by specific metrics.
        """
        metrics_values = {}
        for metric in metrics:
            if metric.metric == mase:
                metrics_values[metric.name] = metric.compute(labels, forecasts, insample=kwargs["input"])
            else:
                metrics_values[metric.name] = metric.compute(labels, forecasts)
        return metrics_values
        
