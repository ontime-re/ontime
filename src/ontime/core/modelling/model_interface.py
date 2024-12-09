from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, NoReturn, Optional
from ..time_series import TimeSeries


class ModelInterface(ABC):
    """
    Abstract class to define methods to implement
    for a Model class inspired by Scikit Learn API.
    """

    @abstractmethod
    def fit(self, ts: TimeSeries, *args, **kwargs) -> NoReturn:
        """Fit a model"""
        pass

    @abstractmethod
    def predict(self, horizon: Any, ts: Optional[TimeSeries] = None, *args, **kwargs) -> Any:
        """Usage of the model to predict values"""
        pass
