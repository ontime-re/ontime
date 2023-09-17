from abc import ABC, abstractmethod
from typing import NoReturn

from ..time_series import TimeSeries


class AbstractBaseProcessor(ABC):
    """Abstract class to define methods to implement
    for a Processor class.
    """

    @abstractmethod
    def process(self, ts: TimeSeries) -> NoReturn:
        """Process time series"""
        raise NotImplementedError
