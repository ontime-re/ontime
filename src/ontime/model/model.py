from darts.models.forecasting.forecasting_model import ModelMeta

from ontime.abstract.abstract_base_model import AbstractBaseModel
from ontime.time_series import TimeSeries

from .libs.darts.forecasting_model import ForecastingModel as DartsForecastingModel
from .libs.skforecast.forecaster_autoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)


class Model(AbstractBaseModel):
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

    def fit(self, ts, **params):
        """
        Fit the model to the given time series
        :param ts: TimeSeries
        :param params: Parameters to pass to the model
        :return: self
        """
        self.model.fit(ts, **params)
        return self

    def predict(self, n, **params):
        """
        Predict n steps into the future
        :param n: Integer
        :param params: Parameters to pass to the predict method
        :return: TimeSeries
        """
        pred = self.model.predict(n, **params)
        return TimeSeries.from_darts(pred)
