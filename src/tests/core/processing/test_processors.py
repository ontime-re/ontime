import unittest

from ontime.core.processing import processors
from ontime.core.processing.processors import Processors


class TestProcessors(unittest.TestCase):
    def test_constructor__creation__should_be_instance_of_processors(self):
        self.assertIsInstance(processors, Processors)

    def test_load__built_in_processors__should_be_available_as_attributes(self):
        for name in ["filler", "mapper", "windower", "correlation", "density"]:
            self.assertIn(name, processors.get_all())
            self.assertTrue(hasattr(processors, name))
