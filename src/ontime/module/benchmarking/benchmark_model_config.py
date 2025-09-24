from __future__ import annotations
from typing import Type, Dict, Any, Callable, Optional
from enum import Enum
import warnings

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
        zero_shot_only: Optional[bool] = None,
        benchmark_mode: Optional[BenchmarkMode] = None,
        static_model_params: Optional[Dict[str, Any]] = None,
        dynamic_model_params: Optional[Dict[str, Callable[[BenchmarkDataset], Any]]] = None,
        dynamic_predict_params: Optional[Dict[str, Callable[[BenchmarkDataset], Any]]] = None,
        dynamic_fit_params: Optional[Dict[str, Callable[[BenchmarkDataset], Any]]] = None,
        validation_set_param: Optional[str] = None,
    ):
        """
        Initializes a BenchmarkModelConfig.

        :param model_name: name of the model
        :param model_class: class of the model to be instanciated
        :param benchmark_mode: DEPRECATED - either zero shot or full shot
        :param static_model_params: dictionary of model parameters that are static, known as soon as the model is declared
        :param dynamic_model_params: dictionary of model parameters that are functions depending on the dataset
        :param dynamic_predict_params: dictionary of additional parameters for model predict method,
        that are functions depending on the dataset
        :param dynamic_fit_params: dictionary of additional parameters for model fit method,
        that are functions depending on the dataset
        :param validation_set_param: name of the parameter for the validation set to give to the model fit method
        :return: the initialized BenchmarkModelConfig
        """
        self.model_name = model_name
        self.zero_shot_only = zero_shot_only
        self.benchmark_mode = benchmark_mode
        self.model_class = model_class
        self.validation_set_param = validation_set_param
        self._static_model_params = static_model_params or {}
        self._dynamic_model_params = dynamic_model_params or {}
        self._dynamic_predict_kwargs = dynamic_predict_params or {}
        self._dynamic_fit_kwargs = dynamic_fit_params or {}


        # depreciation about benchmark_mode
        if benchmark_mode is not None:
            warnings.warn(
                "The 'benchmark_mode' parameter is deprecated and will be removed in a future version. "
                "Use 'zero_shot_only' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if zero_shot_only is None:
                zero_shot_only = benchmark_mode == BenchmarkMode.ZERO_SHOT

        self.zero_shot_only = zero_shot_only if zero_shot_only is not None else False

    @staticmethod
    def _resolve_dynamic_params(
            dynamic_map: Dict[str, Callable[[BenchmarkDataset], Any]],
            dataset: BenchmarkDataset,
    ) -> Dict[str, Any]:
        """
        Helper method to resolve dynamic parameters for a dataset.

        :param dynamic_map: map from dataset name to callable function
        :param dataset: dataset to resolve dynamic parameters for
        :return: the dynamic parameters
        """
        return {key: fn(dataset) for key, fn in dynamic_map.items()}

    def init_model(self, dataset: BenchmarkDataset) -> AbstractModel:
        """
        Initializes the model from its class and parameters. Dynamically computes any parameters that depend on the dataset.

        :param dataset: The dataset being used to compute dynamic parameters.
        :return: the initialized model
        """

        # resolve dynamic parameters that depend on dataset
        resolved_model_dynamic_params = self._resolve_dynamic_params(self._dynamic_model_params, dataset)
        model_params = {**self._static_model_params, **resolved_model_dynamic_params}

        return self.model_class(**model_params)

    def get_predict_kwargs(self, dataset: BenchmarkDataset):
        """
        Returns the resolved dynamic kwargs for model predict method.
        :param dataset: The dataset being used to compute dynamic kwargs.
        :return: the resolved prediction kwargs
        """
        return self._resolve_dynamic_params(self._dynamic_predict_kwargs, dataset)

    def get_fit_kwargs(self, dataset: BenchmarkDataset):
        """
        Returns the resolved dynamic kwargs for model fit method.
        :param dataset: The dataset being used to compute dynamic kwargs.
        :return: the resolved prediction kwargs
        """
        return self._resolve_dynamic_params(self._dynamic_fit_kwargs, dataset)

