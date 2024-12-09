from abc import ABCMeta
from typing import Union, Type, Optional
from sklearn.base import BaseEstimator
from ...model_interface import ModelInterface
from ontime.core.time_series import TimeSeries
from skforecast.ForecasterAutoregMultiVariate import (
    ForecasterAutoregMultiVariate as SkForecasterAutoregMultiVariate,
)


class ForecasterAutoregMultiVariate(ModelInterface):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, model: Union[Type[BaseEstimator], BaseEstimator], **params):
        """
        Initialize model based on ForecasterAutoregMultiVariate. **params are defined in ForecasterAutoregMultiVariate
        from sklearn
        """
        super().__init__()
        # check if model is a class or an instance
        if isinstance(model, type):
            model = model()
        self.model = SkForecasterAutoregMultiVariate(regressor=model, **params)

    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoregMultiVariate":
        """
        Fit model SkForecasterAutoregMultiVariate, params are defined in ForecasterAutoregMultiVariate from sklearn

        :param ts: TimeSeries used to fit the model
        :param params: params used to fit the model documented in ForecasterAutoregMultiVariate from sklearn

        Return the model fitted
        """
        df = ts.pd_dataframe()
        self.model.fit(series=df, **params)
        return self

    def predict(self, n: int, ts: Optional[TimeSeries] = None, **params) -> TimeSeries:
        """
        Predict n steps into the future, must match with the limit defined on model initialization

        :param n: int number of steps to predict must match with the limit defined on model initialization
        :param params: params used to predict the model documented in ForecasterAutoregMultiVariate from sklearn

        Return the prediction in a TimeSeries
        """
        pred = self.model.predict(n, **params)
        return TimeSeries.from_dataframe(pred)
