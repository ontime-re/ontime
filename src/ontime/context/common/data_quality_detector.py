from ...core.time_series import TimeSeries, BinaryTimeSeries
from ...core.detection import Quantile, Threshold


class DataQualityDetector:
    """
    Detects anomalies in a time series given a threshold or quantile.
    """

    def __init__(
        self,
        threshold_type: str,
        upper_threshold: float = None,
        lower_threshold: float = None,
    ):
        """
        Constructor for the DataQualityDetector class.

        :param threshold_type: str, either 'quantile' or 'threshold'
        :param upper_threshold: absolute value of the upper threshold or quantile
        :param lower_threshold: absolute value of the lower threshold or quantile
        """
        self.threshold_type = threshold_type
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.is_fitted = False

    def fit(self, ts: TimeSeries):
        """
        Fits the detector to the given time series.

        :param ts: TimeSeries
        """
        match self.threshold_type:
            case "quantile":
                self.detector = Quantile(
                    low_quantile=self.lower_threshold,
                    high_quantile=self.upper_threshold,
                )
                self.detector.fit(ts)
            case "threshold":
                self.detector = Threshold(
                    low_threshold=self.lower_threshold,
                    high_threshold=self.upper_threshold,
                )
        self.is_fitted = True

    def detect(self, ts: TimeSeries) -> BinaryTimeSeries:
        """
        Detects anomalies in the given time series given a quantile or threshold crossing.

        :param ts: TimeSeries
        :return: BinaryTimeSeries with 0 for normal values and 1 for anomalies
        """
        assert self.is_fitted, "Detector has not been fitted"
        return self.detector.detect(ts)
