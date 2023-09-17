from typing import Optional, Tuple, Union, List

from darts.models.forecasting.tcn_model import TCNModel
import pytorch_lightning as pl

from ...abstract import AbstractBaseModel
from ...time_series import TimeSeries


class TCN(TCNModel, AbstractBaseModel):
    """
    Wrapper around Darts TCN model.
    """

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        kernel_size: int = 3,
        num_filters: int = 3,
        num_layers: Optional[int] = None,
        dilation_base: int = 2,
        weight_norm: bool = False,
        dropout: float = 0.2,
        **kwargs
    ):
        """Temporal Convolutional Network Model (TCN).

        This is an implementation of a dilated TCN used for forecasting, inspired from https://arxiv.org/abs/1803.01271.

        This model supports past covariates (known for `input_chunk_length` points before prediction time).

        Some other arguments can be added to the model by passing them as keyword arguments. To know them, check
        the Darts documentation for the :class:`TCNModel` class.

        input_chunk_length
            Number of past time steps that are fed to the forecasting module.
        output_chunk_length
            Number of time steps the torch module will predict into the future at once.
        kernel_size
            The size of every kernel in a convolutional layer.
        num_filters
            The number of filters in a convolutional layer of the TCN.
        weight_norm
            Boolean value indicating whether to use weight normalization.
        dilation_base
            The base of the exponent that will determine the dilation on every level.
        num_layers
            The number of convolutional layers.
        dropout
            The dropout rate for every convolutional layer. This is compatible with Monte Carlo dropout
            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
            prediction time).
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        """
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            kernel_size,
            num_filters,
            num_layers,
            dilation_base,
            weight_norm,
            dropout,
            **kwargs
        )

    def fit(
        self,
        ts: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        val_series: Optional[TimeSeries] = None,
        val_past_covariates: Optional[TimeSeries] = None,
        val_future_covariates: Optional[TimeSeries] = None,
        trainer: Optional[pl.Trainer] = None,
        verbose: Optional[bool] = None,
        epochs: int = 0,
        max_samples_per_ts: Optional[int] = None,
        num_loader_workers: int = 0,
    ):
        """Fit/train the model on one or multiple series.

        This method wraps around :func:`fit_from_dataset()`, constructing a default training
        dataset for this model. If you need more control on how the series are sliced for training, consider
        calling :func:`fit_from_dataset()` with a custom :class:`darts.utils.data.TrainingDataset`.

        Training is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        This function can be called several times to do some extra training. If ``epochs`` is specified, the model
        will be trained for some (extra) ``epochs`` epochs.

        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try fitting a model with the wrong covariates argument.

        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.

        :params ts: TimeSeries
            A series or sequence of series serving as target (i.e. what the model will be trained to forecast)
        :params past_covariates: Optional[TimeSeries]
            Optionally, a series or sequence of series specifying past-observed covariates
        :params future_covariates: Optional[TimeSeries]
            Optionally, a series or sequence of series specifying future-known covariates
        :params val_series: Optional[TimeSeries]
            Optionally, one or a sequence of validation target series, which will be used to compute the validation
            loss throughout training and keep track of the best performing models.
        :params val_past_covariates: Optional[TimeSeries]
            Optionally, the past covariates corresponding to the validation series (must match ``covariates``)
        :params val_future_covariates: Optional[TimeSeries]
            Optionally, the future covariates corresponding to the validation series (must match ``covariates``)
        :params trainer: Optional[pl.Trainer]
            Optionally, a custom PyTorch-Lightning Trainer object to perform training. Using a custom ``trainer`` will
            override Darts' default trainer.
        :params verbose: Optional[bool]
            Optionally, whether to print progress.
        :params epochs: int
            If specified, will train the model for ``epochs`` (additional) epochs, irrespective of what ``n_epochs``
            was provided to the model constructor.
        :params max_samples_per_ts: Optional[int]
            Optionally, a maximum number of samples to use per time series. Models are trained in a supervised fashion
            by constructing slices of (input, output) examples. On long time series, this can result in unnecessarily
            large number of training samples. This parameter upper-bounds the number of training samples per time
            series (taking only the most recent samples in each series). Leaving to None does not apply any
            upper bound.
        :params num_loader_workers: int
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            both for the training and validation loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.

        :return self:
            Fitted model.
        """
        super().fit(
            ts,
            past_covariates,
            future_covariates,
            val_series,
            val_past_covariates,
            val_future_covariates,
            trainer,
            verbose,
            epochs,
            max_samples_per_ts,
            num_loader_workers,
        )
        return self

    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        trainer: Optional[pl.Trainer] = None,
        batch_size: Optional[int] = None,
        verbose: Optional[bool] = None,
        n_jobs: int = 1,
        roll_size: Optional[int] = None,
        num_samples: int = 1,
        num_loader_workers: int = 0,
        mc_dropout: bool = False,
        predict_likelihood_parameters: bool = False,
    ) -> TimeSeries:
        """Predict the ``n`` time step following the end of the training series, or of the specified ``series``.

        Prediction is performed with a PyTorch Lightning Trainer. It uses a default Trainer object from presets and
        ``pl_trainer_kwargs`` used at model creation. You can also use a custom Trainer with optional parameter
        ``trainer``. For more information on PyTorch Lightning Trainers check out `this link
        <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ .

        Below, all possible parameters are documented, but not all models support all parameters. For instance,
        all the :class:`PastCovariatesTorchModel` support only ``past_covariates`` and not ``future_covariates``.
        Darts will complain if you try calling :func:`predict()` on a model with the wrong covariates argument.

        Darts will also complain if the provided covariates do not have a sufficient time span.
        In general, not all models require the same covariates' time spans:

        * | Models relying on past covariates require the last ``input_chunk_length`` of the ``past_covariates``
          | points to be known at prediction time. For horizon values ``n > output_chunk_length``, these models
          | require at least the next ``n - output_chunk_length`` future values to be known as well.
        * | Models relying on future covariates require the next ``n`` values to be known.
          | In addition (for :class:`DualCovariatesTorchModel` and :class:`MixedCovariatesTorchModel`), they also
          | require the "historic" values of these future covariates (over the past ``input_chunk_length``).

        When handling covariates, Darts will try to use the time axes of the target and the covariates
        to come up with the right time slices. So the covariates can be longer than needed; as long as the time axes
        are correct Darts will handle them correctly. It will also complain if their time span is not sufficient.

        :param n: int
            The number of time steps after the end of the training time series for which to produce predictions
        :param series: Optional[TimeSeries]
            Optionally, a series or sequence of series, representing the history of the target series whose
            future is to be predicted. If specified, the method returns the forecasts of these
            series. Otherwise, the method returns the forecast of the (single) training series.
        :param past_covariates: Optional[TimeSeries]
            Optionally, the past-observed covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        :param future_covariates: Optional[TimeSeries]
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        :param trainer: Optional[pl.Trainer]
            Optionally, a custom PyTorch-Lightning Trainer object to perform prediction. Using a custom ``trainer``
            will override Darts' default trainer.
        :param batch_size: Optional[int]
            Size of batches during prediction. Defaults to the models' training ``batch_size`` value.
        :param verbose: Optional[bool]
            Optionally, whether to print progress.
        :param n_jobs: int
            The number of jobs to run in parallel. ``-1`` means using all processors. Defaults to ``1``.
        :param roll_size: Optional[int]
            For self-consuming predictions, i.e. ``n > output_chunk_length``, determines how many
            outputs of the model are fed back into it at every iteration of feeding the predicted target
            (and optionally future covariates) back into the model. If this parameter is not provided,
            it will be set ``output_chunk_length`` by default.
        :param num_samples: int
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        :param num_loader_workers: int
            Optionally, an integer specifying the ``num_workers`` to use in PyTorch ``DataLoader`` instances,
            for the inference/prediction dataset loaders (if any).
            A larger number of workers can sometimes increase performance, but can also incur extra overheads
            and increase memory usage, as more batches are loaded in parallel.
        :param mc_dropout: bool
            Optionally, enable monte carlo dropout for predictions using neural network based models.
            This allows bayesian approximation by specifying an implicit prior over learned models.
        :param predict_likelihood_parameters: bool
            If set to `True`, the model predict the parameters of its Likelihood parameters instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``

        :return TimeSeries:
            One time series containing the forecasts of ``series``, or the forecast of the training series
            if ``series`` is not specified and the model has been trained on a single series.
        """
        ts_predicted = super().predict(
            n,
            series,
            past_covariates,
            future_covariates,
            trainer,
            batch_size,
            verbose,
            n_jobs,
            roll_size,
            num_samples,
            num_loader_workers,
            mc_dropout,
            predict_likelihood_parameters,
        )
        return TimeSeries.from_darts(ts_predicted)
