from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractBaseProcessor(ABC):
    """Abstract class to define methods to implement
    for a Processor class.
    """

    @abstractmethod
    def process(self, ts) -> NoReturn:
        """Process time series"""
        raise NotImplementedError
