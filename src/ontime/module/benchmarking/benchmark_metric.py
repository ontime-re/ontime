from ontime.core.time_series.time_series import TimeSeries


class BenchmarkMetric:
    def __init__(self, name: str, metric_function):
        self.metric = metric_function
        self.name = name

    def compute(self, target: TimeSeries, pred: TimeSeries, **kwargs):
        """
        Compute the metric on the target and predicted time series.
        """
        return self.metric(target, pred, component_reduction=None, **kwargs)
