from abc import ABCMeta
from ...abstract_model import AbstractModel
from ....time_series import TimeSeries
from skforecast.ForecasterAutoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)


class ForecasterAutoreg(AbstractModel):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model, **params):
        super().__init__()
        if isinstance(model, ABCMeta):
            model = model()
        self.model = SkForecastForecasterAutoreg(regressor=model, **params)

    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoreg":
        self.model.fit(y=ts.pd_series(), **params)
        return self

    def predict(self, n: int, **params) -> TimeSeries:
        pred = self.model.predict(n, **params)
        return TimeSeries.from_series(pred)
