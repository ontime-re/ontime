import unittest

from ontime.core.detection import detectors
from ontime.core.detection.detectors import Detectors


class TestDetectors(unittest.TestCase):
    def test_constructor__creation__should_be_instance_of_detectors(self):
        self.assertIsInstance(detectors, Detectors)

    def test_load__built_in_detectors__should_be_available_as_attributes(self):
        for name in ["threshold", "quantile"]:
            self.assertIn(name, detectors.get_all())
            self.assertTrue(hasattr(detectors, name))
