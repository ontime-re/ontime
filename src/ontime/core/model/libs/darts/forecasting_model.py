from ...abstract_model import AbstractModel
from ....time_series import TimeSeries


class ForecastingModel(AbstractModel):
    """
    Generic wrapper around Darts forecasting models
    """

    def __init__(self, model, **params):
        """Constructor of a ForecastingModel object

        :param model: Dart's forecasting model
        :param params: dict of keyword arguments for this model's constructor
        """
        super().__init__()
        self.model = model(**params)

    def fit(self, ts, **params) -> "ForecastingModel":
        """
        Fit the model to the given time series

        :param ts: TimeSeries
        :param params: dict of keyword arguments for this model's fit method
        :return: self
        """
        self.model.fit(ts, **params)
        return self

    def predict(self, n, **params) -> TimeSeries:
        """
        Predict n steps into the future

        :param n: int number of steps to predict
        :param params: dict of keyword arguments for this model's predict method
        :return: TimeSeries
        """
        pred = self.model.predict(n, **params)
        return TimeSeries.from_darts(pred)
