from ontime.abstract.abstract_base_model import AbstractBaseModel
from ontime.time_series import TimeSeries


class ForecastingModel(AbstractBaseModel):
    """
    Generic wrapper around Darts forecasting models
    """

    def __init__(self, model, **params):
        super().__init__()
        self.model = model(**params)

    def fit(self, ts: TimeSeries, **params) -> "ForecastingModel":
        print("pass")
        self.model.fit(ts, **params)
        return self

    def predict(self, n: int, **params) -> TimeSeries:
        pred = self.model.predict(n, **params)
        return TimeSeries.from_darts(pred)
