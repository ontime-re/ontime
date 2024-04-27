from abc import ABC, abstractmethod
from typing import NoReturn


class AbstractGenerator(ABC):
    """
    Abstract class to define methods to implement for a Generator class.
    """

    @abstractmethod
    def generate(self, *args, **kwargs) -> NoReturn:
        """Generate features"""
        raise NotImplementedError
