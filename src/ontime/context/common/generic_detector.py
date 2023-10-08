from darts.models import CatBoostModel
from darts.utils.statistics import check_seasonality
import ontime as on


class GenericDetector:
    """
    Generic detector for univariate time series using CatBoost model
    """

    def __init__(self):
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
        self.model = on.Model(
            CatBoostModel,
            lags=int(lags),
        )
        self.model.fit(ts)

        # Create detector
        self.detector = on.detectors.quantile(low_quantile=self.low_quantile, high_quantile=self.high_quantile)
        self.detector.fit(ts)

        return self

    def detect(self, ts):
        """
        Detect anomalies in the given time series
        :param ts: TimeSeries
        :return: TimeSeries
        """
        return self.detector.detect(ts)

    def predetect(self, n):
        """
        Predict n steps into the future and detect anomalies
        :param n: Int number of steps to predict
        :return: TimeSeries
        """
        pred = self.model.predict(n)
        return self.detector.detect(pred)