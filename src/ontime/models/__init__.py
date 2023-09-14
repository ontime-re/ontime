from .models import Models
from .registry.arima import ARIMA
from .registry.catboost import CatBoost
from .registry.tcn import TCN

models = Models()
models.load("arima", ARIMA)
models.load("catboost", CatBoost)
models.load("TCN", TCN)
