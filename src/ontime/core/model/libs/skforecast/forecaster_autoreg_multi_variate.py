from abc import ABCMeta
from ...abstract_model import AbstractModel
from ontime.core.time_series import TimeSeries
from skforecast.ForecasterAutoregMultiVariate import (
    ForecasterAutoregMultiVariate as SkForecastForecasterAutoregMultiVariate,
)


class ForecasterAutoregMultiVariate(AbstractModel):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model, **params):
        super().__init__()
        if isinstance(model, ABCMeta):
            model = model()
        self.model = SkForecastForecasterAutoregMultiVariate(regressor=model, **params)

    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoregMultiVariate":
        df = ts.pd_dataframe()
        self.model.fit(series=df, **params)
        return self

    def predict(self, n: int, **params) -> TimeSeries:
        pred = self.model.predict(n, **params)
        return TimeSeries.from_dataframe(pred)
