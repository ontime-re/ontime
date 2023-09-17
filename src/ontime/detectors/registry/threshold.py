from darts.ad.detectors.threshold_detector import ThresholdDetector

from ...abstract import AbstractBaseDetector
from ...time_series import TimeSeries


class Threshold(ThresholdDetector, AbstractBaseDetector):
    """
    Wrapper around Darts ThresholdDetector.
    """

    def __init__(self, low_threshold=None, high_threshold=None):
        super().__init__(low_threshold, high_threshold)

    def detect(self, ts: TimeSeries) -> TimeSeries:
        """
        Detects anomalies in the given time series.
        :param ts: TimeSeries
        :return: TimeSeries
        """
        ts_detected = super().detect(ts)
        return TimeSeries.from_darts(ts_detected)
