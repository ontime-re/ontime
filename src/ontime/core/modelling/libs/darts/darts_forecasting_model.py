from typing import Optional, Union, Type, List
from ...abstract_model import AbstractModel
from ....time_series import TimeSeries
from darts.models.forecasting.forecasting_model import ModelMeta, GlobalForecastingModel
import numpy as np


class DartsForecastingModel(AbstractModel):
    """
    Generic wrapper around Darts forecasting models
    """

    def __init__(self, model: Union[Type[ModelMeta], ModelMeta], **params):
        """Constructor of a ForecastingModel object

        :param model: Darts forecasting model class
        :param params: dict of keyword arguments for this model's constructor
        """
        super().__init__()
        self.model = model
        # check if model is a class or an instance
        if isinstance(model, type):
            self.model = model(**params)

    def fit(self, ts: TimeSeries, **params) -> "DartsForecastingModel":
        self.model.fit(ts, **params)
        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        if ts:
            if isinstance(self.model, GlobalForecastingModel):
                pred = self.model.predict(series=ts, n=n, **params)
            else:
                if isinstance(ts, list):
                    pred = [self._fit_predict(n, t, **params) for t in ts]
                else:
                    pred = self._fit_predict(n, ts, **params)
        else:
            pred = self.model.predict(n, **params)
        if isinstance(pred, list):
            return [TimeSeries.from_darts(p) for p in pred]
        return TimeSeries.from_darts(pred)

    def _fit_predict(self, n: int, ts: TimeSeries, **params) -> TimeSeries:
        """
        For LocalForecastingModel, fit the model on the given time series and produce from it a forecast of a given horizon.

        :param n: int number of steps to predict
        :param ts: the time series from which make the prediction. Optional if the model can predict on the ts it has been trained on.
        :param params: dict of keyword arguments for this model's predict method
        :return: TimeSeries
        """
        predictions = []
        # loop over each component
        for i in range(ts.n_components):
            component = ts.univariate_component(i)
            self.model.fit(component)
            forecast = self.predict(n, **params)
            predictions.append(forecast)
        # combine predictions
        combined_forecast = TimeSeries.from_times_and_values(
            times=predictions[0].time_index,
            values=np.column_stack([f.values() for f in predictions]),
        )
        return combined_forecast.with_columns_renamed(
            combined_forecast.components, ts.components
        )
