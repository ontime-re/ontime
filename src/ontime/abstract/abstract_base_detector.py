from abc import ABC, abstractmethod

from ..time_series import BinaryTimeSeries, TimeSeries


class AbstractBaseDetector(ABC):
    """Abstract class to define methods to implement
    for a Detector class.
    """

    # TODO check if this must return a TimeSeries or a BinaryTimeSeries
    @abstractmethod
    def detect(self, ts: TimeSeries) -> BinaryTimeSeries:
        """Detect features"""
        raise NotImplementedError
