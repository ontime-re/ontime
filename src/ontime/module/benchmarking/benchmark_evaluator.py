from typing import List, Dict, Any

from ontime.core.modelling.model import Model
from ..benchmarking import BenchmarkDataset, BenchmarkMetric
from ontime.module.processing.common import (
    split_in_windows,
    split_inputs_from_targets,
)


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

    def evaluate(self, model: Model, batch_size: int = None) -> Dict[str, Any]:
        """
        Evaluation method, computing metrics for each batch of data, and aggregating it.

        :param model: the model to evaluate
        :param batch_size: size of a batch of data. If not specified, will be equal to the number of samples generated.
        :return: calculated metrics
        """
        # create windows
        window_length = (
            self.dataset.input_length + self.dataset.target_length + self.dataset.gap
        )
        ts_list = split_in_windows(self.test_ts, window_length, self.dataset.stride)
        input_ts_list, target_ts_list = split_inputs_from_targets(
            ts_list,
            input_length=self.dataset.input_length,
            target_length=self.dataset.target_length,
            gap_length=self.dataset.gap,
        )

        # batch prediction, take all input in a one batch if no batch size given
        if batch_size is None:
            batch_size = len(input_ts_list)

        pred_ts_list = []

        for i in range(0, len(input_ts_list), batch_size):
            batch_inputs = input_ts_list[i : i + batch_size]
            pred_ts_list.extend(
                model.predict(ts=batch_inputs, n=self.dataset.target_length)
            )  # model should be able to handle list of inputs

        results = {}

        for metric in self.metrics:
            metric_results = metric.compute(
                target_ts_list, pred_ts_list, insample=input_ts_list
            )
            results[metric.name] = metric.aggregate_series_metrics(metric_results)

        return results
