from abc import ABCMeta
from typing import Union, Type, Optional, List
from sklearn.base import BaseEstimator
from ...abstract_model import AbstractModel
from ....time_series import TimeSeries
from skforecast.ForecasterAutoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)


class ForecasterAutoreg(AbstractModel):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model: Union[Type[BaseEstimator], BaseEstimator], **params):
        """
        Initialize model based on ForecasterAutoreg. **params are defined in ForecasterAutoreg from sklearn

        :param model: Model to wrap
        :param params: Additional parameters to pass to the model
        """
        super().__init__()
        
        # check if model is a class or an instance
        if isinstance(model, type):
            model = model()
        self.model = SkForecastForecasterAutoreg(regressor=model, **params)


    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoreg":
        """
        Fit the model

        :param ts: TimeSeries to fit the model on
        :param params: Additional parameters to pass to the model
        :return: self
        """

        # Fit the model
        self.model.fit(y=ts.pd_series(), **params)

        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        """
        Make a prediction

        :param n: Number of steps to predict
        :param ts: Optional TimeSeries to contain the predicted values
        :param params: Additional parameters to pass to the model
        :return: TimeSeries with the predicted values
        """
        if ts is None:
            pred = self.model.predict(n, **params)
        else:
            if not isinstance(ts, TimeSeries):
                raise ValueError(
                    f"For now, predict method can only be used on single TimeSeries"
                )
            pred = self.model.predict(n, last_window=ts.pd_series(), **params)

        # Create a TimeSeries object from the DataFrame
        ts_pred = TimeSeries.from_series(pred)
        return ts_pred
