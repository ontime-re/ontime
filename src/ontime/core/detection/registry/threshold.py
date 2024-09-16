from typing import Sequence, Union

from darts.ad.detectors.threshold_detector import ThresholdDetector

from ..abstract_detector import AbstractDetector
from ...time_series import TimeSeries, BinaryTimeSeries
from ...utils.anomaly_logger import BinaryAnomalyLogger


class Threshold(ThresholdDetector, AbstractDetector):
    """
    Wrapper around Darts ThresholdDetector.
    """

    def __init__(
        self,
        low_threshold: Union[int, float, None] = None,
        high_threshold: Union[int, float, None] = None,
        enable_logging: bool = False,
        logger_params: dict = None,
    ):
        """
        :param low_threshold: lower bounds
            The lower bound is included in the valid interval. So if the lower bound is 0, the value 0 is valid.

        :param high_threshold: (Sequence of) upper bounds.
            The upper bound is included in the valid interval. So if the upper bound is 10, the value 10 is valid.
        """
        super().__init__(low_threshold, high_threshold)
        self.enable_logging = enable_logging
        default_params = {"description": "ThresholdDetector"}
        self.logger_params = default_params if logger_params is None else logger_params
        if enable_logging:
            self.logger = BinaryAnomalyLogger(**self.logger_params)

    def detect(self, ts: TimeSeries) -> BinaryTimeSeries:
        """
        Detects anomalies in the given time series.

        :param ts: TimeSeries
        :return: BinaryTimeSeries
        """
        ts_detected = super().detect(ts)
        ts_detected = BinaryTimeSeries(ts_detected.data_array())

        if self.enable_logging:
            self.logger.log_anomalies(ts_detected)

        return ts_detected
