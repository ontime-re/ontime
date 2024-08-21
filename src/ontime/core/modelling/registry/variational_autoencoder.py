from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .autoencoder import Encoder, Decoder, Autoencoder
from ontime.core.time_series import TimeSeries


class VariationalEncoder(Encoder):
    def __init__(self, entry_size: int, latent_dims: int):
        super(VariationalEncoder, self).__init__(entry_size, latent_dims)
        self.insize = entry_size
        self.midsize = latent_dims + (entry_size - latent_dims) // 2
        self.outsize = latent_dims
        self.linear1 = nn.Linear(entry_size, self.midsize)
        self.linear2 = nn.Linear(self.midsize, latent_dims)
        self.linear3 = nn.Linear(self.midsize, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0
        self.double()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z

    def __str__(self):
        s = f"Encoder: 3 layers\n\
                layer 0: {self.insize} -> {self.midsize}\n\
                layer 1(mu): {self.midsize} -> {self.outsize}\n\
                layer 2(sigma): {self.midsize} -> {self.outsize}"
        return s


class VariationalAutoencoder(Autoencoder):
    def __init__(self, entry_size: int, latent_dims: int):
        super(VariationalAutoencoder, self).__init__(entry_size, latent_dims)
        self.encoder = VariationalEncoder(entry_size, latent_dims)
        self.decoder = Decoder(latent_dims, entry_size)
        self.double()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def loss(self, x, x_hat):
        return ((x - x_hat) ** 2).sum() + self.encoder.kl

    # train method should be correctly inherited from AutoEncoder

    @staticmethod
    def new_encoder_for_dataset(
        dataset: TimeSeries, period
    ) -> "VariationalAutoencoder":
        entry_size = len(dataset.columns.tolist()) * period
        latent_dims = entry_size // 4  # arbitrary choice
        return VariationalAutoencoder(entry_size, latent_dims)

    def __str__(self):
        return str(self.encoder) + "\n" + str(self.decoder)
