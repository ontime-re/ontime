from darts.models.forecasting.forecasting_model import ModelMeta
from sklearn.base import BaseEstimator
from ..time_series import TimeSeries
from .abstract_model import AbstractModel
from .libs.darts.forecasting_model import ForecastingModel as DartsForecastingModel
from .libs.skforecast.forecaster_autoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)
from .libs.skforecast.forecaster_autoreg_multi_variate import (
    ForecasterAutoregMultiVariate as SkForecasterAutoregMultiSeries,
)


class Model(AbstractModel):
    """
    Generic wrapper around SkForecast and DartsForecast time series libraries

    The model is automatically selected based on the size of the time series.
    It is chosen once and then kept for the whole lifecycle of the model.
    """

    def __init__(self, model, **params):
        super().__init__()
        self.model = model
        self.params = params
        self.is_model_undefined = True

    def fit(self, ts: TimeSeries, **params) -> "Model":
        """
        Fit the model to the given time series

        :param ts: TimeSeries
        :param params: Parameters to pass to the model
        :return: self
        """
        if self.is_model_undefined:
            self._set_model(ts)
            self.is_model_undefined = False

        self.model.fit(ts, **params)
        return self

    def predict(self, n: int, **params) -> TimeSeries:
        """
        Predict length steps into the future

        :param n: int number of steps to predict
        :param params: dict to pass to the predict method
        :return: TimeSeries
        """
        pred = self.model.predict(n, **params)
        return TimeSeries.from_darts(pred)

    def _set_model(self, ts):
        size_of_ts = ts.n_components
        if issubclass(self.model.__class__, ModelMeta):
            # Darts Models
            self.model = DartsForecastingModel(self.model, **self.params)
        # This take all the sklearn regressors and pipelines
        elif issubclass(self.model, BaseEstimator):
            if size_of_ts > 1:
                # scikit-learn API compatible models
                self.model = SkForecasterAutoregMultiSeries(self.model, **self.params)
            else:
                # scikit-learn API compatible models
                self.model = SkForecastForecasterAutoreg(self.model, **self.params)
        else:
            raise ValueError(
                f"The {self.model} Model is not supported by the Model wrapper."
            )
