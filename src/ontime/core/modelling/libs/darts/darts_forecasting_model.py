from typing import Optional
from ...abstract_model import AbstractModel
from ....time_series import TimeSeries
from darts.models.forecasting.forecasting_model import ModelMeta, GlobalForecastingModel
import numpy as np


class DartsForecastingModel(AbstractModel):
    """
    Generic wrapper around Darts forecasting models
    """

    def __init__(self, model_class: ModelMeta, **params):
        """Constructor of a ForecastingModel object

        :param model: Darts forecasting model class
        :param params: dict of keyword arguments for this model's constructor
        """
        super().__init__()
        self.model = model_class(**params)

    def fit(self, ts: TimeSeries, **params) -> "DartsForecastingModel":
        """
        Fit the model to the given time series

        :param ts: TimeSeries
        :param params: dict of keyword arguments for this model's fit method
        :return: self
        """
        self.model.fit(ts, **params) # TODO: should we not here remove **params so that we can pass any args to fit method from outside.
        return self

    def predict(self, n: int, ts: Optional[TimeSeries] = None, **params) -> TimeSeries:
        """
        Predict n steps into the future

        :param n: int number of steps to predict
        :param ts: the time series from which make the prediction. Optional if the model can predict on the ts it has been trained on.
        :param params: dict of keyword arguments for this model's predict method
        :return: TimeSeries
        """
        if ts:
            if not isinstance(self.model, GlobalForecastingModel):
                pred = self._fit_predict(ts, n, **params)
            pred = self.model.predict(series=ts, n=n, **params)
        else:            
            pred = self.model.predict(n, **params)
        return TimeSeries(pred.data_array()) # why ?
    
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
        return combined_forecast.with_columns_renamed(combined_forecast.components, ts.components)
