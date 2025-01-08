from abc import ABCMeta
from typing import Union, Type, Optional, List
from sklearn.base import BaseEstimator
from ...abstract_model import AbstractModel
from ontime.core.time_series import TimeSeries
from skforecast.ForecasterAutoregMultiVariate import (
    ForecasterAutoregMultiVariate as SkForecasterAutoregMultiVariate,
)


class ForecasterAutoregMultiVariate(AbstractModel):
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
        self.model.fit(series=ts.pd_dataframe(), **params)
        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        if ts is None:
            pred = self.model.predict(n, **params)
        else:
            if not isinstance(ts, TimeSeries):
                raise ValueError(
                    f"For now, predict method can only be used on single TimeSeries"
                )
            pred = self.model.predict(n, last_window=ts.pd_dataframe(), **params)
        return TimeSeries.from_dataframe(pred)
