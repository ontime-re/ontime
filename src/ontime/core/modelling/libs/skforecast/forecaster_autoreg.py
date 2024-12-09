from abc import ABCMeta
from typing import Union, Type, Optional
from sklearn.base import BaseEstimator
from ...model_interface import ModelInterface
from ....time_series import TimeSeries
from skforecast.ForecasterAutoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)


class ForecasterAutoreg(ModelInterface):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model: Union[Type[BaseEstimator], BaseEstimator], **params):
        """
        Initialize model based on ForecasterAutoreg. **params are defined in ForecasterAutoreg from sklearn
        """
        super().__init__()
        # check if model is a class or an instance
        if isinstance(model, type):
            model = model()
        self.model = SkForecastForecasterAutoreg(regressor=model, **params)

    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoreg":
        """
        Fit model SkForecasterAutoreg, params are defined in ForecasterAutoreg from sklearn

        :param ts: TimeSeries used to fit the model
        :param params: params used to fit the model documented in ForecasterAutoreg from sklearn

        Return the model fitted
        """
        self.model.fit(y=ts.pd_series(), **params)
        return self

    def predict(self, n: int, ts: Optional[TimeSeries] = None, **params) -> TimeSeries:
        """
        Predict n steps into the future, must match with the limit defined on model initialization

        :param n: int number of steps to predict
        :param params: params used to predict the model documented in ForecasterAutoreg from sklearn

        Return the prediction in a TimeSeries
        """
        pred = self.model.predict(n, **params)
        return TimeSeries.from_series(pred)
