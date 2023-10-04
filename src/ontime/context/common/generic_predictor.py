from darts.models import CatBoostModel
from darts.utils.statistics import check_seasonality
import ontime as on


class GenericPredictor:
    """
    Generic predictor for univariate time series using CatBoost model
    """

    def __init__(self):
        self.model = None

    def fit(self, ts):
        """
        Fit the model to the given time series
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
        return self

    def predict(self, n):
        """
        Predict n steps into the future
        :param n: Int number of steps to predict
        :return: TimeSeries
        """
        return self.model.predict(n)
