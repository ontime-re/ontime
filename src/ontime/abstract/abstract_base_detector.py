from abc import ABC, abstractmethod

from ..time_series import TimeSeries


class AbstractBaseDetector(ABC):
    """Abstract class to define methods to implement
    for a Detector class.
    """

    @abstractmethod
    def detect(self, ts: TimeSeries) -> TimeSeries:
        """Detect features"""
        raise NotImplementedError
