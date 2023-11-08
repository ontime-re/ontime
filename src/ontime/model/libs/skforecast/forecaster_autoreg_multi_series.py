from abc import ABCMeta

from ontime.abstract.abstract_base_model import AbstractBaseModel
from ontime.time_series import TimeSeries

from skforecast.ForecasterAutoregMultiSeries import (
    ForecasterAutoregMultiSeries as SkForecastForecasterAutoregMultiSeries,
)


class ForecasterAutoregMultiSeries(AbstractBaseModel):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model, **params):
        super().__init__()
        if isinstance(model, ABCMeta):
            model = model()
        self.model = SkForecastForecasterAutoregMultiSeries(regressor=model, **params)

    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoregMultiSeries":
        df = ts.pd_dataframe()
        self.model.fit(series=df, **params)
        return self

    def predict(self, n: int, **params) -> TimeSeries:
        pred = self.model.predict(n, **params)
        return TimeSeries.from_dataframe(pred)
