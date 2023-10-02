from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, NoReturn

from ..time_series import TimeSeries


class AbstractBaseModel(ABC):
    """Abstract class to define methods to implement
    for a Model class inspired by Scikit Learn API.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, ts: TimeSeries) -> NoReturn:
        """Fit a model"""
        pass

    @abstractmethod
    def predict(self, horizon: Any) -> Any:
        """Usage of the model to predict values"""
        pass
