from abc import ABCMeta

from ontime.abstract.abstract_base_model import AbstractBaseModel
from ontime.time_series import TimeSeries

from skforecast.ForecasterAutoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)


class ForecasterAutoreg(AbstractBaseModel):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model, **params):
        super().__init__()
        if isinstance(model, ABCMeta):
            model = model()
        self.model = SkForecastForecasterAutoreg(regressor=model, **params)

    def fit(self, ts, **params):
        self.model.fit(y=ts.pd_series(), **params)
        return self

    def predict(self, n, **params):
        pred = self.model.predict(n, **params)
        return TimeSeries.from_series(pred)
