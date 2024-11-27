from ontime.core.time_series.time_series import TimeSeries
import numpy as np


class BenchmarkMetric:
    def __init__(self, name: str, metric_function, component_reduction=np.nanmean, series_reduction=np.nanmean):
        self.metric = metric_function
        self.name = name
        self.component_reduction = component_reduction
        self.series_reduction = series_reduction

    def compute(self, target: TimeSeries, pred: TimeSeries, **kwargs):
        """
        Compute the metric on the target and predicted time series.
        """
        return self.metric(target, pred, component_reduction=self.component_reduction, series_reduction=self.series_reduction, **kwargs)
