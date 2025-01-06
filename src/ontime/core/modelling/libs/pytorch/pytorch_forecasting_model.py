from typing import Any, Optional, Dict, Union, Type, List
import warnings

from ...abstract_model import AbstractModel
from ....time_series import TimeSeries
from ontime.module.processing.pytorch.time_series_data_module import (
    TimeSeriesDataModule,
)

import lightning as L
import torch
from torch import nn
import pandas as pd


class TorchForecastingModel(L.LightningModule, AbstractModel):
    """
    Generic wrapper around PyTorch forecasting model, using PyTorch lightning for training.
    """

    def __init__(
        self,
        model: Union[Type[nn.Module], nn.Module],
        input_chunk_length: int,
        output_chunk_length: int,
        n_epochs: int = 10,
        loss_fn=nn.MSELoss(),
        lr: float = 0.001,
        train_data_module_params: Dict[str, Any] = {},
        **params,
    ):
        """Constructor of a TorchForecastingModel object

        :param model: a torch model class (not instantiated)
        :param input_chunk_length: number of time steps in the past the model use for making one predicton
        :param output_chunk_length: number of time steps to be predicted by the model at once
        :param n_epochs: number of training epochs
        :param loss_fn: loss function for the model
        :param lr: learning rate for the optimizer
        :param train_data_module_params: params for the training data module to use. Input length and target length are defined by input_chunk_length and output_chunk_length parameters
        """
        super(TorchForecastingModel, self).__init__()
        self.model = model
        # check if model is a class or an instance
        if isinstance(model, type):
            self.model = model(**params)
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.lr = lr
        self.data_module_params = train_data_module_params
        self.train_ts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, ts: TimeSeries, **params) -> "TorchForecastingModel":
        self.train_ts = ts
        trainer = L.Trainer(max_epochs=self.n_epochs, **params)
        self.data_module = TimeSeriesDataModule(
            series=ts,
            input_length=self.input_chunk_length,
            target_length=self.output_chunk_length,
            **self.data_module_params,
        )
        trainer.fit(self, datamodule=self.data_module)
        return self

    def predict(
        self, n: int, ts: Optional[Union[List[TimeSeries], TimeSeries]] = None, **params
    ) -> Union[List[TimeSeries], TimeSeries]:
        if not ts:
            ts = self.train_ts

        is_batch = isinstance(ts, list)
        ts_list = ts if is_batch else [ts]

        if n > self.output_chunk_length:
            warnings.warn(
                f"The requested prediction horizon (n={n}) exceeds the model's output_chunk_length "
                f"({self.output_chunk_length}). The model will use an iterative forecasting approach, "
                f"which may result in reduced accuracy due to error propagation.",
            )

        # create forecast indices
        forecast_indices = [
            pd.date_range(
                start=series.time_index[-1], periods=n + 1, freq=series.time_index.freq
            )[1:]
            for series in ts_list
        ]

        # prepare inputs
        input = torch.stack([series.to_tensor() for series in ts_list])

        self.eval()
        with torch.no_grad():
            # as the model may not be designed to predict n time steps at once, we may slice the output and/or autoregressively get predictions
            predictions = []
            current_input = input.clone()

            for _ in range(0, n, self.output_chunk_length):
                output = self(current_input)

                predictions.append(output)

                # use current prediction in next input
                current_input = torch.cat((current_input, output), dim=1)

            # concatenate predictions and trim
            predictions = torch.cat(predictions, dim=1)
            predictions = predictions[:, :n, :]

        result = [
            TimeSeries.from_times_and_values(
                forecast_index, prediction.squeeze(0).cpu().numpy()
            )
            for forecast_index, prediction in zip(forecast_indices, predictions)
        ]

        return result if is_batch else result[0]
