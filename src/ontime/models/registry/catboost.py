from typing import Optional, Tuple, Union, List

from darts.models.forecasting.catboost_model import CatBoostModel

from ...abstract import AbstractBaseModel
from ...time_series import TimeSeries


class CatBoost(CatBoostModel, AbstractBaseModel):
    """
    Wrapper around Darts CatBoost model.
    """

    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        add_encoders: Optional[dict] = None,
        likelihood: str = None,
        quantiles: List = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """
        CatBoost Model

        :param lags:
             Lagged target values used to predict the next time step. If an integer is given the last `lags` past lags
            are used (from -1 backward). Otherwise a list of integers with lags is required (each lag must be < 0).
        :param lags_past_covariates:
            Number of lagged past_covariates values used to predict the next time step. If an integer is given the last
            `lags_past_covariates` past lags are used (inclusive, starting from lag -1). Otherwise a list of integers
            with lags < 0 is required.
        :param lags_future_covariates:
            Number of lagged future_covariates values used to predict the next time step. If an tuple (past, future) is
            given the last `past` lags in the past are used (inclusive, starting from lag -1) along with the first
            `future` future lags (starting from 0 - the prediction time - up to `future - 1` included). Otherwise a list
            of integers with lags is required.
        :param output_chunk_length:
            Number of time steps predicted at once by the internal regression model. Does not have to equal the forecast
            horizon `n` used in `predict()`. However, setting `output_chunk_length` equal to the forecast horizon may
            be useful if the covariates don't extend far enough into the future.
        :param add_encoders:
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:
                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
        :param likelihood:
            Can be set to 'quantile', 'poisson' or 'gaussian'. If set, the model will be probabilistic,
            allowing sampling at prediction time. When set to 'gaussian', the model will use CatBoost's
            'RMSEWithUncertainty' loss function. When using this loss function, CatBoost returns a mean
            and variance couple, which capture data (aleatoric) uncertainty.
            This will overwrite any `objective` parameter.
        :param quantiles:
            Fit the model to these quantiles if the `likelihood` is set to `quantile`.
        :param random_state:
            Control the randomness in the fitting procedure and for sampling.
            Default: ``None``.
        :param multi_models:
            If True, a separate model will be trained for each future lag to predict. If False, a single model is
            trained to predict at step 'output_chunk_length' in the future. Default: True.
        :param use_static_covariates:
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        :param kwargs:
            Additional keyword arguments passed to `catboost.CatBoostRegressor`.
        """
        super().__init__(
            lags,
            lags_past_covariates,
            lags_future_covariates,
            output_chunk_length,
            add_encoders,
            likelihood,
            quantiles,
            random_state,
            multi_models,
            use_static_covariates,
            **kwargs,
        )

    def fit(
        self,
        ts: TimeSeries,
        past_covariates: TimeSeries = None,
        future_covariates: Optional[TimeSeries] = None,
        val_series: Optional[TimeSeries] = None,
        val_past_covariates: Optional[TimeSeries] = None,
        val_future_covariates: Optional[TimeSeries] = None,
        max_samples_per_ts: Optional[int] = None,
        verbose: Optional[Union[int, bool]] = 0,
        **kwargs,
    ):
        """Fits/trains the model using the provided list of features time series and the target time series.

        :param ts:
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        :param past_covariates:
            Optionally, a series or sequence of series specifying past-observed covariates
        :param future_covariates:
            Optionally, a series or sequence of series specifying future-known covariates
        :param val_series:
            TimeSeries or Sequence[TimeSeries] object containing the target values for evaluation dataset
        :param val_past_covariates:
            Optionally, a series or sequence of series specifying past-observed covariates for evaluation dataset
        :param val_future_covariates : Union[TimeSeries, Sequence[TimeSeries]]
            Optionally, a series or sequence of series specifying future-known covariates for evaluation dataset
        :param max_samples_per_ts:
            This is an integer upper bound on the number of tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        :param verbose:
            An integer or a boolean that can be set to 1 to display catboost's default verbose output
        :param **kwargs:
            Additional kwargs passed to `catboost.CatboostRegressor.fit()`
        :return: self
        """
        super().fit(
            ts,
            past_covariates,
            future_covariates,
            val_series,
            val_past_covariates,
            val_future_covariates,
            max_samples_per_ts,
            verbose,
            **kwargs,
        )
        return self

    def predict(
        self,
        n: int,
        ts: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
        **kwargs,
    ) -> TimeSeries:
        """Forecasts values for `n` time steps after the end of the series.

        :param n: int
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        :param ts: TimeSeries, optional
            Optionally, one or several input `TimeSeries`, representing the history of the target series whose future
            is to be predicted. If specified, the method returns the forecasts of these series. Otherwise, the method
            returns the forecast of the (single) training series.
        :param past_covariates: TimeSeries or list of TimeSeries, optional
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension and type.
        :param future_covariates: TimeSeries or list of TimeSeries, optional
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension and type.
        :param num_samples: int, default: 1
            Number of times a prediction is sampled from a probabilistic model. Should be set to 1
            for deterministic models.
        :param verbose:
            Optionally, whether to print progress.
        :param predict_likelihood_parameters:
            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``
        :param **kwargs : dict, optional
            Additional keyword arguments passed to the `predict` method of the model. Only works with
            univariate target series.
        """
        ts_predicted = super().predict(
            n,
            ts,
            past_covariates,
            future_covariates,
            num_samples,
            verbose,
            predict_likelihood_parameters,
            **kwargs,
        )
        return TimeSeries.from_darts(ts_predicted)
