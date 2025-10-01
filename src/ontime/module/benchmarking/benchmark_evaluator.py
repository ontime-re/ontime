from typing import List, Dict, Any, Union, Tuple

from ontime import TimeSeries
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

    def __init__(
        self,
        dataset: BenchmarkDataset,
        metrics: List[BenchmarkMetric],
        on_val_ts: bool = False,
    ):
        """
        Initializes a BenchmarkEvaluator

        :param dataset: dataset on which evaluate models
        :param metrics: evaluation metrics to compute
        :param on_val_ts: if True, use the validation time series for evaluation, instead of the test time series, default to False
        :return: an initialized BenchmarkEvaluator
        """
        self.dataset = dataset
        self.metrics = metrics
        if on_val_ts:
            _, self.test_ts = dataset.get_train_val_split()
        else:
            _, self.test_ts = dataset.get_train_test_split()

    def evaluate(
        self,
        model: Model,
        scaler: Scaler = None,
        return_predictions: bool = False,
        scaled_evaluation: bool = False,
        predict_kwargs: Dict[str, Any] = None,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], List[TimeSeries]]]:
        """
        Evaluation method, computing metrics for each batch of data, and aggregating it.

        :param model: the model to evaluate
        :param scaler: scaler to use for scaling the time series, default to None
        :param return_predictions: if True, return the predictions as well, default to False
        :param scaled_evaluation: whether to compute metrics and predictions on scaled data, default to False
        :param predict_kwargs: additional arguments to pass to model.predict, default to None
        :return: calculated metrics
        """
        if predict_kwargs is None:
            predict_kwargs = {}

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

        if scaler is not None:
            input_ts_list = [scaler.transform(ts) for ts in input_ts_list]

        batch_size = self.dataset.test_batch_size

        pred_ts_list = []

        for i in range(0, len(input_ts_list), batch_size):
            batch_inputs = input_ts_list[i : i + batch_size]
            pred_ts_list.extend(
                model.predict(
                    ts=batch_inputs, n=self.dataset.target_length, **predict_kwargs
                )
            )  # model should be able to handle list of inputs

        if scaler is not None:
            if scaled_evaluation:
                # transform the target as well
                target_ts_list = [scaler.transform(ts) for ts in target_ts_list]
            else:
                # inverse transform the predictions and the input time series (for insample)
                pred_ts_list = [scaler.inverse_transform(ts) for ts in pred_ts_list]
                input_ts_list = [scaler.inverse_transform(ts) for ts in input_ts_list]

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

        return (results, pred_target_ts_list) if return_predictions else results
