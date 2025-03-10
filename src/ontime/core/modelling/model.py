from typing import Optional, Union, Type, Any, List
from darts.models.forecasting.forecasting_model import ForecastingModel
from sklearn.base import BaseEstimator
from ..time_series import TimeSeries
from .abstract_model import AbstractModel
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


class Model(AbstractModel):
    """
    Generic wrapper around time series libraries

    At the moment the following libraries are supported :

    - onTime models
    - Darts models
    - Scikit-learn models

    The model is automatically selected based on the size of the time series.
    It is chosen once and then kept for the whole lifecycle of the model.
    """

    def __init__(self, model: Union[AbstractModel, Type[AbstractModel]], **params):
        """
        Initializes a Model.

        :param model: either a model class or a model instance
        """

        super().__init__()
        self.model = model
        self.params = params
        self.is_model_undefined = True
        self.index_name = None
        self.components_names = None

    def fit(self, ts: TimeSeries, **params) -> "Model":
        """
        Fit the model

        :param ts: TimeSeries to fit the model on
        :param params: Additional parameters to pass to the model
        :return: self
        """
        # Set the model if it is undefined
        if self.is_model_undefined:
            self._set_model(ts)
        
        # Initialize index name and components names
        self.index_name = ts.time_index.name
        self.components_names = ts.components.to_list()

        # Fit the model
        self.model.fit(ts, **params)
        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        """
        Make a prediction

        :param n: Number of steps to predict
        :param ts: Optional TimeSeries to contain the predicted values
        :param params: Additional parameters to pass to the model
        :return: TimeSeries with the predicted values
        """
        # Set the model if it is undefined
        if self.is_model_undefined:
            if isinstance(ts, list):
                self._set_model(ts[0])
            else:
                self._set_model(ts)

        # Make the prediction
        ts_pred = self.model.predict(n, ts, **params)

        # Set the index name and components names
        ts_pred.time_index.name = self.index_name
        rename_dict = dict(zip(ts_pred.components.to_list(), self.components_names))
        ts_pred = ts_pred.rename(rename_dict)

        return ts_pred

    def _set_model(self, ts: TimeSeries):
        """
        Create and set the appropriate model wrapper according to the actual model.

        :param ts: the time series on which the selected model will be fitted.
        :raises ValueError: if the provided model is not supported by the model wrapper. This could happen if the model
        does not inherit from a known base class such as `ModelMeta`, `BaseEstimator`, or `nn.Module`.
        """

        size_of_ts = ts.n_components

        if is_subclass_or_instance_of_subclass(self.model, ForecastingModel):
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
