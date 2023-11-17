from typing import Sequence, Union

from darts.ad.detectors.threshold_detector import ThresholdDetector

from ..abstract_detector import AbstractDetector
from ...time_series import TimeSeries, BinaryTimeSeries


class Threshold(ThresholdDetector, AbstractDetector):
    """
    Wrapper around Darts ThresholdDetector.
    """

    def __init__(
        self,
        low_threshold: Union[int, float, Sequence[float], None] = None,
        high_threshold: Union[int, float, Sequence[float], None] = None,
    ):
        """

        :param low_threshold: (Sequence of) lower bounds.
            If a sequence, must match the dimensionality of the series
            The lower bound is included in the valid interval. So if the lower bound is 0, the value 0 is valid.

        :param high_threshold: (Sequence of) upper bounds.
            If a sequence, must match the dimensionality of the series
            The upper bound is included in the valid interval. So if the upper bound is 10, the value 10 is valid.
        """
        super().__init__(low_threshold, high_threshold)

    def detect(self, ts: TimeSeries) -> BinaryTimeSeries:
        """
        Detects anomalies in the given time series.
        :param ts: TimeSeries
        :return: BinaryTimeSeries
        """
        ts_detected = super().detect(ts)
        return BinaryTimeSeries.from_darts(ts_detected)
