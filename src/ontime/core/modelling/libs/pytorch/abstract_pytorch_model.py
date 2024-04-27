from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, NoReturn

import pytorch_lightning as pl

from ontime.core.time_series import TimeSeries


class AbstractPytorchModel(ABC, pl.LightningModule):
    """
    Abstract class to define methods to implement
    for a Model class inspired by Scikit Learn API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def fit(self, ts: TimeSeries, *args, **kwargs) -> NoReturn:
        """Fit a model"""
        pass

    @abstractmethod
    def predict(self, horizon: Any, *args, **kwargs) -> Any:
        """Usage of the model to predict values"""
        pass
