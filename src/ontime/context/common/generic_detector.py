from darts.models import CatBoostModel
from darts.utils.statistics import check_seasonality
from ...time_series import BinaryTimeSeries
from ...detectors import Quantile
from ...model import Model


class GenericDetector:
    """
    Generic detector for univariate time series using CatBoost model
    """

    def __init__(self):
        self.detector = None
        self.model = None
        self.low_quantile = 0.2
        self.high_quantile = 0.8

    def fit(self, ts):
        """
        Fit the model to the given time series as well as a detector
        :param ts: TimeSeries
        :return: self
        """
        # Get information about the time series
        has_seasonality, seasonality = check_seasonality(ts)
        lags = 12 if seasonality == 0 else seasonality

        # Create model
        self.model = Model(
            CatBoostModel,
            lags=int(lags),
        )
        self.model.fit(ts)

        # Create detector
        self.detector = Quantile(
            low_quantile=self.low_quantile, high_quantile=self.high_quantile
        )
        self.detector.fit(ts)

        return self

    def detect(self, ts) -> BinaryTimeSeries:
        """
        Detect anomalies in the given time series
        :param ts: TimeSeries
        :return: BinaryTimeSeries
        """
        if self.detector is None:
            raise ValueError("Detector has not been fitted")
        return self.detector.detect(ts)

    def predetect(self, n) -> BinaryTimeSeries:
        """
        Predict n steps into the future and detect anomalies

        Can raise a ValueError if the model has not been fitted

        :param n: Int number of steps to predict
        :return: BinaryTimeSeries
        """
        if self.model is None:
            raise ValueError("Model has not been fitted")
        pred = self.model.predict(n)
        return self.detector.detect(pred)
