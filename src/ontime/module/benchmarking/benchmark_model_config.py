from __future__ import annotations
from typing import Type
from enum import Enum

from ontime.core.modelling.model_interface import ModelInterface

class BenchmarkMode(Enum):
    ZERO_SHOT = 1  # no training, only inference
    FULL_SHOT = 3  # full training

class BenchmarkModelConfig:
    """
    BenchmarkModelConfig class that holds model class and its configuration to be instanciated and used in a benchmark.
    """
    def __init__(self, model_name: str, model_class: Type[ModelInterface], benchmark_mode: BenchmarkMode, **model_params):
        """
        Initializes a BenchmarkModelConfig.

        :param model_name: name of the model
        :param model_class: class of the model to be instanciated
        :param benchmark_mode: either zero shot or full shot
        :return: the initialized BenchmarkModelConfig
        """
        self.model_name = model_name
        self.benchmark_mode = benchmark_mode
        self._model_class = model_class
        self._model_params = model_params

    def init_model(self) -> ModelInterface:
        """
        Initializes the model from its class and parameters

        :return: the initialized model
        """
        
        return self._model_class(**self._model_params)
