from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch import Tensor

from ontime.core.time_series import TimeSeries
from ontime.module.processing.pytorch.sliced_dataset import SlicedDataset

device = "cpu"


class Encoder(nn.Module):
    def __init__(self, entry_size: int, latent_dims: int):
        super(Encoder, self).__init__()
        self.insize = entry_size
        self.midsize = latent_dims + (entry_size - latent_dims) // 2
        self.outsize = latent_dims
        self.linear1 = nn.Linear(entry_size, self.midsize)
        self.linear2 = nn.Linear(self.midsize, latent_dims)
        self.double()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def __str__(self):
        s = f"Encoder: 2 layers\n\
                layer 0: {self.insize} -> {self.midsize}\n\
                layer 1: {self.midsize} -> {self.outsize}\n\
                "
        return s


class Decoder(nn.Module):
    def __init__(self, latent_dims: int, output_size: int):
        super(Decoder, self).__init__()
        self.insize = latent_dims
        self.midsize = latent_dims + (output_size - latent_dims) // 2
        self.outsize = output_size
        self.linear1 = nn.Linear(latent_dims, self.midsize)
        self.linear2 = nn.Linear(self.midsize, output_size)
        self.double()

    def forward(self, z: Tensor) -> Tensor:
        z = F.relu(self.linear1(z))
        z = self.linear2(z)
        return z

    def __str__(self):
        s = f"Decoder: 2 layers\n\
                layer 0: {self.insize} -> {self.midsize}\n\
                layer 1: {self.midsize} -> {self.outsize}\n\
                "
        return s


class Autoencoder(nn.Module):
    def __init__(self, entry_size: int, latent_dims: int):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(entry_size, latent_dims)
        self.decoder = Decoder(latent_dims, entry_size)
        self.double()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def loss(self, x: Tensor, x_hat: Tensor) -> float:
        loss = F.smooth_l1_loss(x, x_hat, reduction="mean")
        return loss

    def get_reconstructed(
        self,
        dataset: TimeSeries,
        period: int,
        labels: TimeSeries = None,
        verbose: bool = False,
    ) -> list[list]:
        ds = SlicedDataset(dataset, period, labels)
        data = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        results_x = []
        results_xhat = []
        results_y = []
        results_loss = []

        for x, y in data:
            x = x.to(device)  # CPU
            x_hat = self(x)
            if verbose:
                print(f"x:{x.size()} -> x_hat {x_hat.size()}")
            x_hat = x_hat.reshape(x.size()).to("cpu").detach()
            loss = self.loss(x, x_hat)
            x_hat = x_hat.numpy()
            x = x.cpu().numpy()
            loss = loss.detach().cpu().numpy().item()
            results_x.append(x)
            results_xhat.append(x_hat)
            results_loss.append(loss)
            if ds.labels is not None:
                results_y.append(y)
        reconstructed_dataset = Autoencoder._into_timeseries(
            dataset, results_xhat, results_loss, results_y, period
        )
        return reconstructed_dataset

    def train(self, data: SlicedDataset, device: str, epochs: int = 20):
        opt = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            i = 1
            for x, y in data:
                i += 1
                x = x.to(device)  # CPU
                opt.zero_grad()
                x_hat = self(x)
                x_hat = torch.reshape(x_hat, x.size())
                loss = self.loss(x, x_hat)
                loss.backward()
                opt.step()

    @staticmethod
    def get_period(dataset: TimeSeries):
        periods = []
        for col in dataset.columns:
            periods.append(
                pyd.findfrequency(dataset.pd_dataframe()[col].to_numpy(), detrend=True)
            )
        period = max(periods)
        while period < 15:
            period += period
        return period

    @staticmethod
    def new_encoder_for_dataset(
        dataset: TimeSeries, period: int = None
    ) -> "Autoencoder":
        if period is None:
            periods = []
            for col in dataset.columns:
                periods.append(
                    pyd.findfrequency(
                        dataset.pd_dataframe()[col].to_numpy(), detrend=True
                    )
                )
            period = max(periods)
            while period < 15:
                period += period
        entry_size = len(dataset.columns.tolist()) * period
        latent_dims = entry_size // 4  # arbitrary choice
        return Autoencoder(entry_size, latent_dims), period

    def __str__(self):
        return str(self.encoder) + "\n" + str(self.decoder)

    @staticmethod
    def _into_timeseries(input_dataset, xhat, loss, y, period):
        # flatten x_hat
        x_hat_flat = []
        for i in range(0, len(xhat)):
            sample = xhat[i][0]
            for line in sample:
                x_hat_flat.append(line)
        reconstructed_dataset = pd.DataFrame(
            x_hat_flat, columns=input_dataset.columns.tolist()
        )
        reconstructed_dataset.index = input_dataset.time_index[
            : len(reconstructed_dataset.index)
        ]

        # loss and y
        reconstructed_loss = []
        reconstructed_y = []
        for l in loss:
            for i in range(0, period):
                reconstructed_loss.append(l)
        reconstructed_dataset["loss"] = reconstructed_loss

        if y is not None and len(y) > 0:
            for ry in y:
                for i in range(0, period):
                    reconstructed_y.append(ry)
            reconstructed_dataset["y"] = reconstructed_y

        return TimeSeries.from_pandas(reconstructed_dataset)
