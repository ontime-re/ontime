from typing import Optional, Union, Sequence, Callable, List
import numpy as np
from darts.metrics import mase
import inspect

from ontime.core.time_series.time_series import TimeSeries


class BenchmarkMetric:
    """
    A Benchmark metric, wrapping darts metrics
    """

    def __init__(
        self,
        name: str,
        metric_function: Callable,
        component_reduction: Optional[Callable] = np.nanmean,
        series_reduction: Callable = np.nanmean,
        accept_nan: bool = True,
    ):
        """
        Initializes a BenchmarkMetric

        :param name: the name of the metric
        :param metric_function: the darts metric function
        :param component_reduction: a function to aggregate the metrics over the component axis. Can be None if no component reduction.
        :param series_reduction: a function to aggregate the metrics over many series.
        :param accept_nan: either to accept returning undefined metric (nan value(s)) when it can not be computed. It
        allows to still compute the metric when many samples are given while metric cannot be computed for some of them. In this case,
        component and series reductions must be able to be calculated with nan values.
        """
        self.name = name
        self._metric_function = metric_function
        self._component_reduction = component_reduction
        self._series_reduction = series_reduction
        self._accept_nan = accept_nan
        self.insample_required = (
            True
            if "insample" in inspect.signature(metric_function).parameters
            else False
        )

    def _compute_with_nan_single(
        self,
        target: TimeSeries,
        pred: TimeSeries,
        insample: Optional[TimeSeries],
        **kwargs,
    ) -> Union[np.ndarray, float]:
        """
        Compute the metric on the target and predicted time series when the series stricly contain only one sample. If
        metric can be computed, return NaN.

        :param target: the true target time series
        :param pred: the predicted time series
        :param insample: the training series used to forecast predicted series, used to compute metrics such as the MASE
        :param kwargs: other key word arguments
        :return: the metric(s)
        """

        if insample is not None:
            kwargs["insample"] = insample
        try:
            return self._metric_function(
                target, pred, component_reduction=self._component_reduction, **kwargs
            )
        except ValueError:
            if self._component_reduction is None:
                return np.full(pred.n_components, np.nan)
            else:
                return np.nan

    def _compute_with_nan(
        self,
        target: Union[TimeSeries, Sequence[TimeSeries]],
        pred: Union[TimeSeries, Sequence[TimeSeries]],
        insample: Optional[Union[TimeSeries, Sequence[TimeSeries]]],
        **kwargs,
    ) -> Union[List[Union[np.ndarray, float]], np.ndarray, float]:
        """
        Compute the metric on the target and predicted time series. If metric is not defined for some samples, it ignores them.

        :param target: the true target time series
        :param pred: the predicted time series
        :param insample: the training series used to forecast predicted series, used to compute metrics such as the MASE
        :param kwargs: other key word arguments
        :return: the metric(s)
        """

        metrics = []
        if isinstance(pred, Sequence):
            if len(pred) != len(target) or (
                insample is not None and len(pred) != len(insample)
            ):
                raise ValueError(
                    "Lengths of pred, target, and insample must match when insample is provided; pred and target "
                    "must match otherwise."
                )
            for i in range(0, len(pred), 1):
                current_insample = insample if insample is None else insample[i]
                metrics.append(
                    self._compute_with_nan_single(
                        target[i], pred[i], current_insample, **kwargs
                    )
                )
            return metrics
        return self._compute_with_nan_single(target, pred, insample, **kwargs)

    def compute(
        self,
        target: Union[TimeSeries, Sequence[TimeSeries]],
        pred: Union[TimeSeries, Sequence[TimeSeries]],
        insample: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        **kwargs,
    ) -> Union[List[Union[np.ndarray, float]], np.ndarray, float]:
        """
        Compute the metric on the target and predicted time series.

        :param target: the true target time series
        :param pred: the predicted time series
        :param insample: the training series used to forecast predicted series, used to compute metrics such as the MASE
        :param kwargs: other key word arguments
        :return: the metric(s)
        """

        if not self.insample_required:
            insample = None
        if self.insample_required and insample is None:
            raise ValueError(
                f"The 'insample' parameter is required with {self._metric_function.__name__}."
            )

        if self._accept_nan:
            return self._compute_with_nan(target, pred, insample)
        else:
            if insample is not None:
                kwargs["insample"] = insample
            return self._metric_function(
                target,
                pred,
                component_reduction=self._component_reduction,
                series_reduction=None,
                **kwargs,
            )

    def aggregate_series_metrics(
        self, metrics: List[Union[np.ndarray, float]]
    ) -> Union[np.ndarray, float]:
        """
        Aggregate metric over given series, according to the reduction function of the metric.

        :param metrics: metrics to aggregate
        return:
        """

        if self._series_reduction is None:
            return metrics
        return self._series_reduction(metrics, axis=0)
