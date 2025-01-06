from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, NoReturn, Optional, Union, List
from ..time_series import TimeSeries


class AbstractModel(ABC):
    """
    Abstract class to define methods to implement
    for a Model class inspired by Scikit Learn API.
    """

    @abstractmethod
    def fit(self, ts: TimeSeries, **params) -> "AbstractModel":
        """
        Fit a model.

        :param ts: time series on which to fit the model
        :return: self
        """
        pass

    @abstractmethod
    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        """
        Predict n steps into the future

        :param n: int number of steps to predict
        :param ts: the time series from which make the prediction. Optional if the model
        can predict on the ts it has been trained on.
        :return: TimeSeries
        """
        pass
