from darts.ad.detectors.quantile_detector import QuantileDetector
from ...abstract import AbstractBaseDetector


class Quantile(QuantileDetector, AbstractBaseDetector):
    def __init__(self, low_quantile=None, high_quantile=None):
        super().__init__(low_quantile, high_quantile)
