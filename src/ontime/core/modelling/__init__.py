from .model import Model
from .models import Models
from .registry.gru import GRU

# from .registry.autoencoder import Autoencoder
# from .registry.variational_autoencoder import VariationalAutoencoder

models = Models()
models.load("gru", GRU)
# models.load("autoencoder", AutoEncoder)
# models.load("variational_autoencoder", VariationalAutoencoder)

__all__ = ["Model", "models"]
