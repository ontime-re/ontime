from ...core.time_series import TimeSeries, BinaryTimeSeries


class MissingDataDetector:
    """
    Detects missing data in a time series.
    """

    def detect(self, ts: TimeSeries) -> BinaryTimeSeries:
        """
        Detects presence of missing data

        :param ts: TimeSeries

        :return: BinaryTimeSeries with 0 for normal values and 1 for anomalies
        """
        return BinaryTimeSeries.from_series(ts.pd_series().isna())
