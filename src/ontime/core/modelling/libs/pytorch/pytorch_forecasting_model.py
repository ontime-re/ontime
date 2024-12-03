from typing import Any, NoReturn, Optional

from ...abstract_model import AbstractModel
from ....time_series import TimeSeries

import pytorch_lightning as pl
import torch
from torch import nn

class TorchForecastingModel(AbstractModel, pl.LightningModule):
    """
    Generic wrapper around PyTorch forecasting model, using PyTorch lightning for training.
    """

    def __init__(self, model_class: nn.Module, loss_fn=nn.MSELoss(), lr: float = 0.001, n_epochs: int = 10, train_batch_size: int = 32, **params):
        """Constructor of a TorchForecastingModel object

        :param model_class: a torch model class (not instantiated)
        :param loss_fn: loss function for the model
        :param lr: learning rate for the optimizer
        :param n_epochs: number of epochs for training 
        :param train_batch_size: batch size for training
        """
        super().__init__()
        self.model = model_class(**params)
        self.loss_fn = loss_fn
        self.lr = lr
        self.n_epochs = n_epochs # TODO: should we not put that in fit method ?
        self.train_batch_size = train_batch_size # TODO: should we not put that in fit method ?
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the wrapped PyTorch model

        :param x: input tensor
        :return: output tensor
        """
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning

        :param batch: tuple containing inputs and targets
        :param batch_idx: index of the batch
        :return: training loss for the batch
        """
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training

        :return: optimizer
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, ts: TimeSeries, **kwargs) -> NoReturn:
        """Fit the model using the given time series data

        :param ts: time series data for training
        :param trainer_kwargs: additional arguments for the PyTorch Lightning trainer
        """
        trainer = pl.Trainer(max_epochs=self.n_epochs, **kwargs) # TODO: should we not here remove **params so that we can pass any args to fit method from outside.
        # TODO : create a dataloader or a lightning data module, but with what info ?
        trainer.fit()

    def predict(self, n: int, ts: Optional[TimeSeries] = None, **params) -> TimeSeries:
        """
        Predict n steps into the future

        :param n: int number of steps to predict
        :param ts: the time series from which make the prediction. Optional if the model can predict on the ts it has been trained on.
        :param params: dict of keyword arguments for this model's predict method
        :return: TimeSeries
        """
        self.eval()
        with torch.no_grad():
            if ts is not None:
                inputs = ts.to_tensor()
            else:
                raise ValueError("Time series data must be provided for prediction.")

            predictions = []
            # TODO: make a prediction based on what the model has been trained on
            # predictions = ...

            return predictions
