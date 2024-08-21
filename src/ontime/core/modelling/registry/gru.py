import torch
from torch import nn
import pytorch_lightning as pl

from ontime.core.modelling.libs.pytorch.abstract_pytorch_model import (
    AbstractPytorchModel,
)


class GRU(AbstractPytorchModel):
    def __init__(self, input_dim, hidden_dim, output_steps, num_layers=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim, output_steps)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])  # taking the last output for prediction
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def fit(self, ts, *args, **kwargs):
        # Assuming ts can be directly loaded as DataLoader
        trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=20)
        trainer.fit(self, ts)

    def predict(self, horizon, *args, **kwargs):
        # here horizon could be used to define the shape of the input tensor
        # For demonstration, just predict using a zero tensor assuming batch size 1
        dummy_input = torch.zeros(1, horizon, self.gru.input_size).to(self.device)
        self.eval()
        with torch.no_grad():
            return self(dummy_input)


# Example usage
# model = TimeSeriesGRU(input_dim=10, hidden_dim=20, output_steps=5)
# ts_loader = YourDataLoader(time_series)
# model.fit(ts_loader)
# predictions = model.predict(horizon=5)
