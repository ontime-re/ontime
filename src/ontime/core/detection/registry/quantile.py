from darts.ad.detectors.quantile_detector import QuantileDetector

from ..abstract_detector import AbstractDetector
from ...time_series import BinaryTimeSeries, TimeSeries
from ...utils.anomaly_logger import BinaryAnomalyLogger


class Quantile(QuantileDetector, AbstractDetector):
    """
    Wrapper around Darts QuantileDetector.
    """

    def __init__(
        self,
        low_quantile: float = None,
        high_quantile: float = None,
        enable_logging: bool = False,
        logger_params: dict = None,
    ):
        """
        Constructor for QuantileDetector

        :param low_quantile: lower quantile
        :param high_quantile: higher quantile
        :param enable_logging:
        """
        super().__init__(low_quantile, high_quantile)
        self.enable_logging = enable_logging
        default_params = {"description": "QuantileDetector"}
        self.logger_params = default_params if logger_params is None else logger_params
        if enable_logging:
            self.logger = BinaryAnomalyLogger(**self.logger_params)

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
        ts_detected = BinaryTimeSeries(ts_detected.data_array())

        if self.enable_logging:
            self.logger.log_anomalies(ts_detected)

        return ts_detected
