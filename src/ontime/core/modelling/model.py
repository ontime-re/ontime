from typing import Optional, Union, Type, Any
from darts.models.forecasting.forecasting_model import ModelMeta
from sklearn.base import BaseEstimator
from ..time_series import TimeSeries
from .model_interface import ModelInterface
from .libs.darts.darts_forecasting_model import DartsForecastingModel
from .libs.skforecast.forecaster_autoreg import (
    ForecasterAutoreg as SkForecastForecasterAutoreg,
)
from .libs.skforecast.forecaster_autoreg_multi_variate import (
    ForecasterAutoregMultiVariate as SkForecasterAutoregMultiSeries,
)
from .libs.pytorch.pytorch_forecasting_model import TorchForecastingModel
from torch import nn

def is_subclass_or_instance_of_subclass(variable: Any, base_class: Any):
    """
    Check if the given variable is either a subclass of the given base class, or an instance of the base class (or its subclass)
    """
    if isinstance(variable, base_class):
        return True 
    try:
        if issubclass(variable, base_class):
            return True
    except TypeError:
        pass
    return False


class Model(ModelInterface):
    """
    Generic wrapper around time series libraries

    At the moment the following libraries are supported :

    - onTime models
    - Darts models
    - Scikit-learn models

    The model is automatically selected based on the size of the time series.
    It is chosen once and then kept for the whole lifecycle of the model.
    """

    def __init__(self, model: Union[ModelInterface, Type[ModelInterface]], **params):
        super().__init__()
        """
        Constructor
        :param model: either a model class or a model instance
        :param params: argument given to the wrapper model constructor
        """
        self.model = model
        self.params = params
        self.is_model_undefined = True

    def fit(self, ts: TimeSeries, **params) -> "Model":
        """
        Fit the model to the given time series

        :param ts: TimeSeries
        :param params: Parameters to pass to the model
        :return: self
        """
        if self.is_model_undefined:
            self._set_model(ts)

        self.model.fit(ts, **params)
        return self

    def predict(self, n: int, ts: Optional[TimeSeries] = None, **params) -> TimeSeries:
        """
        Predict length steps into the future

        :param n: int number of steps to predict
        :param ts: the time series from which make the prediction. Optional if the model can predict on the ts it has been trained on.
        :param params: dict to pass to the predict method
        :return: TimeSeries
        """
        return self.model.predict(n, ts, **params)

    def _set_model(self, ts):
        """
        Create and set the appropriate model wrapper according to the actual model.
        """
        size_of_ts = ts.n_components

        if is_subclass_or_instance_of_subclass(self.model, ModelMeta):
            # Darts Models
            self.model = DartsForecastingModel(self.model, **self.params)
        # This take all the sklearn regressors and pipelines
        elif is_subclass_or_instance_of_subclass(self.model, BaseEstimator):
            if size_of_ts > 1:
                # scikit-learn API compatible models
                self.model = SkForecasterAutoregMultiSeries(self.model, **self.params)
            else:
                # scikit-learn API compatible models
                self.model = SkForecastForecasterAutoreg(self.model, **self.params)
        elif is_subclass_or_instance_of_subclass(self.model, nn.Module):
            self.model = TorchForecastingModel(self.model, **self.params)
        else:
            raise ValueError(
                f"The {self.model} Model is not supported by the Model wrapper."
            )
        self.is_model_undefined = False
