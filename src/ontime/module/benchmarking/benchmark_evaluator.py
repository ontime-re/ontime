from typing import List, Dict, Any

from ontime.core.modelling.model import Model
from ..benchmarking import BenchmarkDataset, BenchmarkMetric
from ontime.module.processing.common import (
    split_in_windows,
    split_inputs_from_targets,
)
from darts.dataprocessing.transformers import Scaler

class BenchmarkEvaluator:
    """
    Evaluator class to benchmark models on a specific dataset, according to different metrics.
    """

    def __init__(self, dataset: BenchmarkDataset, metrics: List[BenchmarkMetric]):
        """
        Initializes a BenchmarkEvaluator

        :param dataset: dataset on which evaluate models
        :param metrics: evaluation metrics to compute
        :return: an initialized BenchmarkEvaluator
        """
        self.dataset = dataset
        self.metrics = metrics
        _, self.test_ts = dataset.get_train_test_split()

    def evaluate(self, model: Model, scaler: Scaler = None) -> Dict[str, Any]:
        """
        Evaluation method, computing metrics for each batch of data, and aggregating it.

        :param model: the model to evaluate
        :param scaler: scaler to use for scaling the time series, default to None
        :return: calculated metrics
        """
        # create windows
        window_length = (
            self.dataset.input_length + self.dataset.target_length + self.dataset.gap
        )

        test_ts = self.test_ts
        if scaler is not None:
            test_ts = scaler.transform(self.test_ts)

        ts_list = split_in_windows(
            test_ts,
            window_length, 
            self.dataset.stride
        )

        input_ts_list, target_ts_list = split_inputs_from_targets(
            ts_list,
            input_length=self.dataset.input_length,
            target_length=self.dataset.target_length,
            gap_length=self.dataset.gap,
        )

        batch_size = self.dataset.test_batch_size

        pred_ts_list = []

        for i in range(0, len(input_ts_list), batch_size):
            batch_inputs = input_ts_list[i : i + batch_size]
            pred_ts_list.extend(
                model.predict(ts=batch_inputs, n=self.dataset.target_length)
            )  # model should be able to handle list of inputs

        if scaler is not None:
            # inverse transform the predictions, we need to do it time series by time series
            pred_ts_list = [
                scaler.inverse_transform(ts) for ts in pred_ts_list
            ]

        # filter target_ts_list to only include the target columns
        target_ts_list = [
            ts.drop_columns(self.dataset.get_input_columns()) for ts in target_ts_list
        ]
        input_target_ts_list = [
            ts.drop_columns(self.dataset.get_input_columns()) for ts in input_ts_list
        ]  # needed for insample
        # keep target components of prediction
        pred_target_ts_list = [
            ts.with_columns_renamed(ts.columns, self.test_ts.columns).drop_columns(
                self.dataset.get_input_columns()
            )
            for ts in pred_ts_list
        ]

        results = {}

        for metric in self.metrics:
            metric_results = metric.compute(
                target_ts_list, pred_target_ts_list, insample=input_target_ts_list
            )
            results[metric.name] = metric.aggregate_series_metrics(metric_results)

        return results
