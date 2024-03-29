from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ontime.core.time_series import TimeSeries
from ontime.core.model.sliced_dataset import SlicedDataset
from torch import Tensor

device = 'cpu'

class Encoder(nn.Module):
    def __init__(self, entry_size: int, latent_dims: int):
        super(Encoder, self).__init__()
        self.insize = entry_size
        self.midsize = latent_dims + (entry_size - latent_dims)//2
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
        self.midsize = latent_dims + (output_size - latent_dims)//2
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

    def loss(self, x: float, x_hat: float) -> float:
        return ((x - x_hat)**2).sum()

    def get_reconstructed(self, dataset: TimeSeries, period: int, labels: TimeSeries = None, verbose: bool = False) -> list[list]:
        ds = SlicedDataset(dataset, period, labels)
        data = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False)
        results_x = []
        results_xhat = []
        results_y = []
        results_loss = []
        
        for x, y in data:
            x = x.to(device) # CPU
            x_hat = self(x)
            if verbose: 
                print(f'x:{x.size()} -> x_hat {x_hat.size()}')
            x_hat = x_hat.reshape(x.size()).to('cpu').detach().numpy()
            loss = self.loss(x, x_hat)
            x = x.cpu().numpy()
            loss = loss.detach().cpu().numpy().item()
            results_x.append(x)
            results_xhat.append(x_hat)
            results_loss.append(loss)
            if ds.labels is not None:
                results_y.append(y)
        if ds.labels is not None:
            return [results_x, results_xhat, results_loss, results_y]
        return [results_x, results_xhat, results_loss]

    def train(self, data: SlicedDataset, device: str, epochs: int = 20):
        opt = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            for x, y in data:
                x = x.to(device) # CPU
                opt.zero_grad()
                x_hat = self(x)
                x_hat = torch.reshape(x_hat, x.size())
                loss = self.loss(x, x_hat)
                loss.backward()
                opt.step()
    
    @staticmethod
    def new_encoder_for_dataset(dataset: TimeSeries, period: int) -> 'Autoencoder':
        entry_size = len(dataset.columns.tolist())*period
        latent_dims = entry_size//4 # arbitrary choice
        return Autoencoder(entry_size, latent_dims)

    def __str__(self):
        return str(self.encoder) + "\n" + str(self.decoder)