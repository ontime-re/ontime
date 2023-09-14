from .detectors import Detectors
from .registry.threshold import Threshold
from .registry.quantile import Quantile

detectors = Detectors()
detectors.load("threshold", Threshold)
detectors.load("quantile", Quantile)
