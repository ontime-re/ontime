from darts.ad.detectors.quantile_detector import QuantileDetector

from ..abstract_detector import AbstractDetector
from ...time_series import BinaryTimeSeries, TimeSeries


class Quantile(QuantileDetector, AbstractDetector):
    """
    Wrapper around Darts QuantileDetector.
    """

    def __init__(self, low_quantile=None, high_quantile=None):
        super().__init__(low_quantile, high_quantile)

    def fit(self, ts: TimeSeries) -> None:
        """
        Fits the detector to the given time series.
        :param ts: TimeSeries
        :return: None
        """
        super().fit(ts)

    def detect(self, ts: TimeSeries) -> BinaryTimeSeries:
        """
        Detects anomalies in the given time series.
        :param ts: TimeSeries
        :return: BinaryTimeSeries
        """
        ts_detected = super().detect(ts)
        return BinaryTimeSeries(ts_detected.data_array())
