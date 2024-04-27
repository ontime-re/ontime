from abc import ABC, abstractmethod

from ..time_series import BinaryTimeSeries, TimeSeries


class AbstractDetector(ABC):
    """
    Abstract class to define methods to implement for a Detector class.
    """

    @abstractmethod
    def detect(self, ts: TimeSeries, *args, **kwargs) -> BinaryTimeSeries:
        """Detect features"""
        raise NotImplementedError
