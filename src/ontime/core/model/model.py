from darts.models.forecasting.forecasting_model import ModelMeta

from ..time_series import TimeSeries
from .abstract_model import AbstractModel
from .libs.darts.forecasting_model import ForecastingModel as DartsForecastingModel
from .libs.skforecast.forecaster_autoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)


class Model(AbstractModel):
    """
    Generic wrapper around all implemented time series libraries
    """

    def __init__(self, model, **params):
        super().__init__()
        if isinstance(model, ModelMeta):
            # Darts Models
            self.model = DartsForecastingModel(model, **params)
        else:
            # scikit-learn API compatible models
            self.model = SkForecastForecasterAutoreg(model, **params)

    def fit(self, ts: TimeSeries, **params):
        """
        Fit the model to the given time series

        :param ts: TimeSeries
        :param params: Parameters to pass to the model
        :return: self
        """
        self.model.fit(ts, **params)
        return self

    def predict(self, n: int, **params):
        """
        Predict length steps into the future

        :param n: int number of steps to predict
        :param params: dict to pass to the predict method
        :return: TimeSeries
        """
        pred = self.model.predict(n, **params)
        return TimeSeries.from_darts(pred)
