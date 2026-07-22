from typing import Union, Type, Optional, List
from sklearn.base import BaseEstimator
from ...abstract_model import AbstractModel
from ...utils import normalize_prediction
from ....time_series import TimeSeries
from skforecast.recursive import (
    ForecasterRecursive as SKForecastForecasterRecursive,
    ForecasterRecursiveMultiSeries as SKForecastForecasterRecursiveMultiSeries,
)


class SkForecastForecastingModel(AbstractModel):
    """
    Generic wrapper around SkForecast recursive forecasters.

    Handles both univariate and multivariate time series: the underlying
    skforecast forecaster (ForecasterRecursive or ForecasterRecursiveMultiSeries)
    is selected at fit time based on the number of components of the series.
    """

    def __init__(self, sk_model: Union[Type[BaseEstimator], BaseEstimator], **params):
        """
        Initialize model based on a scikit-learn compatible estimator.
        **params are forwarded to the skforecast forecaster constructor.
        """
        super().__init__()
        # check if model is a class or an instance
        if isinstance(sk_model, type):
            sk_model = sk_model()
        self.sk_model = sk_model
        self.params = params
        self.model = None
        self.is_multivariate = False
        self.train_ts = None

    def fit(self, ts: TimeSeries, **params) -> "SkForecastForecastingModel":
        self.is_multivariate = ts.n_components > 1
        self.train_ts = ts
        if self.is_multivariate:
            self.model = SKForecastForecasterRecursiveMultiSeries(
                estimator=self.sk_model, **self.params
            )
            self.model.fit(series=ts.pd_dataframe(), **params)
        else:
            self.model = SKForecastForecasterRecursive(
                estimator=self.sk_model, **self.params
            )
            self.model.fit(y=ts.pd_series(), **params)
        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        if self.model is None:
            raise ValueError("The model must be fitted before calling predict")
        if ts is None:
            pred = self.model.predict(n, **params)
            reference = self.train_ts
        else:
            if not isinstance(ts, TimeSeries):
                raise ValueError(
                    f"For now, predict method can only be used on single TimeSeries"
                )
            last_window = ts.pd_dataframe() if self.is_multivariate else ts.pd_series()
            pred = self.model.predict(n, last_window=last_window, **params)
            reference = ts
        if self.is_multivariate:
            # skforecast returns predictions in long format (level, pred); pivot to wide
            if "level" in pred.columns:
                pred = pred.pivot(columns="level", values="pred")
            pred = TimeSeries.from_dataframe(pred)
        else:
            pred = TimeSeries.from_series(pred)
        if reference is not None:
            pred = normalize_prediction(pred, reference)
        return pred
