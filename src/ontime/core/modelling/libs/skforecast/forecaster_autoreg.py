from abc import ABCMeta
from typing import Union, Type, Optional, List
from sklearn.base import BaseEstimator
from ...abstract_model import AbstractModel
from ...utils import normalize_prediction
from ....time_series import TimeSeries
from skforecast.recursive import (
    ForecasterRecursive as SKForecastForecasterRecursive,
)


class ForecasterAutoreg(AbstractModel):
    """
    Generic wrapper around SkForecast ForecasterAutoreg models
    """

    def __init__(self, sk_model: Union[Type[BaseEstimator], BaseEstimator], **params):
        """
        Initialize model based on ForecasterAutoreg. **params are defined in ForecasterAutoreg from sklearn
        """
        super().__init__()
        # check if model is a class or an instance
        if isinstance(sk_model, type):
            sk_model = sk_model()
        self.model = SKForecastForecasterRecursive(estimator=sk_model, **params)
        self.train_ts = None

    def fit(self, ts: TimeSeries, **params) -> "ForecasterAutoreg":
        self.train_ts = ts
        self.model.fit(y=ts.pd_series(), **params)
        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        if ts is None:
            pred = self.model.predict(n, **params)
            reference = self.train_ts
        else:
            if not isinstance(ts, TimeSeries):
                raise ValueError(
                    f"For now, predict method can only be used on single TimeSeries"
                )
            pred = self.model.predict(n, last_window=ts.pd_series(), **params)
            reference = ts
        pred = TimeSeries.from_series(pred)
        if reference is not None:
            pred = normalize_prediction(pred, reference)
        return pred
