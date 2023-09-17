from typing import Sequence, Union

from darts.ad.detectors.quantile_detector import QuantileDetector

from ...abstract import AbstractBaseDetector
from ...time_series import TimeSeries


class Quantile(QuantileDetector, AbstractBaseDetector):
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

    def detect(self, ts: TimeSeries) -> TimeSeries:
        """
        Detects anomalies in the given time series.
        :param ts: TimeSeries
        :return: TimeSeries
        """
        ts_detected = super().detect(ts)
        return TimeSeries.from_darts(ts_detected)
