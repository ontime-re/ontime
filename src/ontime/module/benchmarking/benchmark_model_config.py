from __future__ import annotations
from typing import Type, Dict, Any, Callable, Optional
from enum import Enum

from ontime.core.modelling.abstract_model import AbstractModel
from .benchmark_dataset import BenchmarkDataset


class BenchmarkMode(Enum):
    ZERO_SHOT = 1  # no training, only inference
    FULL_SHOT = 3  # full training


class BenchmarkModelConfig:
    """
    BenchmarkModelConfig class that holds model class and its configuration to be instanciated and used in a benchmark.
    """

    def __init__(
        self,
        model_name: str,
        model_class: Type[AbstractModel],
        benchmark_mode: BenchmarkMode,
        static_model_params: Optional[Dict[str, Any]] = None,
        dynamic_model_params: Optional[
            Dict[str, Callable[[BenchmarkDataset], Any]]
        ] = None,
        validation_set_param: Optional[str] = None,
    ):
        """
        Initializes a BenchmarkModelConfig.

        :param model_name: name of the model
        :param model_class: class of the model to be instanciated
        :param benchmark_mode: either zero shot or full shot
        :param static_model_params: dictionnary of model parameters that are static, known as soon as the model is declared
        :param static_model_params: dictionnary of model parameters that are functions depending on the dataset
        :param validation_set_param: name of the parameter for the validation set to give to the model fit method
        :return: the initialized BenchmarkModelConfig
        """
        self.model_name = model_name
        self.benchmark_mode = benchmark_mode
        self.model_class = model_class
        self.validation_set_param = validation_set_param
        self._static_model_params = static_model_params or {}
        self._dynamic_model_params = dynamic_model_params or {}

    def init_model(self, dataset: BenchmarkDataset) -> AbstractModel:
        """
        Initializes the model from its class and parameters. Dynamically computes any parameters that depend on the dataset.

        :param dataset: The dataset being used to compute dynamic parameters.
        :return: the initialized model
        """

        # resolve dynamic parameters that depend on dataset
        resolved_model_dynamic_params = {
            key: func(dataset) for key, func in self._dynamic_model_params.items()
        }
        model_params = {**self._static_model_params, **resolved_model_dynamic_params}

        return self.model_class(**model_params)
