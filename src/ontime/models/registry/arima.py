from typing import Optional, Tuple

from darts.models.forecasting.arima import ARIMA

from ...abstract import AbstractBaseModel
from ...time_series import TimeSeries


class ARIMA(ARIMA, AbstractBaseModel):
    """
    Wrapper around Darts ARIMA model.
    """

    def __init__(
        self,
        p: int = 12,
        d: int = 1,
        q: int = 0,
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        random_state: Optional[int] = None,
        add_encoders: Optional[dict] = None,
    ):
        """
        ARIMA

        ARIMA-type models extensible with exogenous variables (future covariates)
        and seasonal components.

        :param p: int
            Order (number of time lags) of the autoregressive model (AR).
        :param d: int
            The order of differentiation; i.e., the number of times the data
            have had past values subtracted (I).
        :param q : int
            The size of the moving average window (MA).
        :param seasonal_order: Tuple[int, int, int, int]
            The (P,D,Q,s) order of the seasonal component for the AR parameters,
            differences, MA parameters and periodicity.
        :param trend: str
            Parameter controlling the deterministic trend. 'n' indicates no trend,
            'c' a constant term, 't' linear trend in time, and 'ct' includes both.
            Default is 'c' for models without integration, and no trend for models with integration.
        :param add_encoders
            A large number of future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'future': ['relative']},
                    'custom': {'future': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
        """
        super().__init__(p, d, q, seasonal_order, trend, random_state, add_encoders)

    def fit(
        self, ts: TimeSeries, future_covariates: Optional[TimeSeries] = None
    ) -> ARIMA:
        """Fit/train the model on the (single) provided series.

        Optionally, a future covariates series can be provided as well.

        :param ts: TimeSeries
            The model will be trained to forecast this time series. Can be multivariate if the model supports it.
        :param future_covariates: TimeSeries
            A time series of future-known covariates. This time series will not be forecasted, but can be used by
            some models as an input. It must contain at least the same time steps/indices as the target `series`.
            If it is longer than necessary, it will be automatically trimmed.
        :return: self
        """
        super().fit(ts, future_covariates)
        return self

    def predict(
        self,
        n: int,
        ts: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> TimeSeries:
        """If the `series` parameter is not set, forecasts values for `n` time steps after the end of the training
        series. If some future covariates were specified during the training, they must also be specified here.

        If the `series` parameter is set, forecasts values for `n` time steps after the end of the new target
        series. If some future covariates were specified during the training, they must also be specified here.

        :param n: int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        :param ts: TimeSeries
            Optionally, a new target series whose future values will be predicted. Defaults to `None`, meaning that the
            model will forecast the future value of the training series.
        :param future_covariates: TimeSeries
            The time series of future-known covariates which can be fed as input to the model. It must correspond to
            the covariate time series that has been used with the :func:`fit()` method for training.

            If `series` is not set, it must contain at least the next `n` time steps/indices after the end of the
            training target series. If `series` is set, it must contain at least the time steps/indices corresponding
            to the new target series (historic future covariates), plus the next `n` time steps/indices after the end.
        :param num_samples: int
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        :return: TimeSeries
        """
        ts_predicted = super().predict(n, ts, future_covariates, num_samples, **kwargs)
        return TimeSeries.from_darts(ts_predicted)
