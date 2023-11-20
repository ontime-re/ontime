from abc import ABC, abstractmethod

from ..time_series import TimeSeries


class AbstractProcessor(ABC):
    """Abstract class to define methods to implement
    for a Processor class.
    """

    @abstractmethod
    def process(self, ts: TimeSeries) -> TimeSeries:
        """Process time series"""
        raise NotImplementedError
