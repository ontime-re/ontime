from ...time_series import BinaryTimeSeries, ProbabilisticTimeSeries, TimeSeries


class AnomaliesFrequencies:
    """
    Class for computing the frequency of anomalies in a time window.
    """

    def __init__(self, anomalies_ts: BinaryTimeSeries):
        self.anomalies_series = anomalies_ts.pd_series()
        pass

    def get_number_of_anomaly_in_window(self, window_size: str) -> TimeSeries:
        """
        Compute the number of anomalies in a time window.

        return: TimeSeries with the number of anomalies in the window
        """
        sum_series = self.anomalies_series.rolling(window=window_size).sum()
        return TimeSeries.from_series(sum_series)

    def get_frequency_of_anomaly_in_window(
        self, window_size: str
    ) -> ProbabilisticTimeSeries:
        """
        Compute the frequency of anomalies in a time window. The frequency is computed as the number of anomalies in
        the window divided by the maximum number of anomalies in a window. So 1 mean that all samples in the window are
        anomalies.

        return: ProbabilisticTimeSeries with the frequency of anomalies in the window
        """
        # Compute the maximum number of anomalies in a window
        max_anomalies = self.anomalies_series.rolling(window=window_size).count().max()
        # Make the aggregation of the anomalies in the window
        sum_series = self.anomalies_series.rolling(window=window_size).sum()
        # Compute the frequency of anomalies in the window
        frequency_series = sum_series / max_anomalies
        return ProbabilisticTimeSeries.from_series(frequency_series)
