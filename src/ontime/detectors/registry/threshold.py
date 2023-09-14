from darts.ad.detectors.threshold_detector import ThresholdDetector
from ...abstract import AbstractBaseDetector


class Threshold(ThresholdDetector, AbstractBaseDetector):
    def __init__(self, low_threshold=None, high_threshold=None):
        super().__init__(low_threshold, high_threshold)
