from ontime.core.time_series.time_series import TimeSeries


class BenchmarkMetric:
    def __init__(self, name: str, metric_function, reduction=None):
        self.metric = metric_function
        self.name = name
        self.reduction = reduction

    def compute(self, target: TimeSeries, pred: TimeSeries):
        """
        Compute the metric on the target and predicted time series.
        """
        return self.metric(target, pred, component_reduction=self.reduction)
