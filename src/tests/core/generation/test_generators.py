import unittest

from ontime.core.generation import generators
from ontime.core.generation.generators import Generators


class TestGenerators(unittest.TestCase):
    def test_constructor__creation__should_be_instance_of_generators(self):
        self.assertIsInstance(generators, Generators)

    def test_load__built_in_generators__should_be_available_as_attributes(self):
        for name in [
            "constant",
            "gaussian",
            "holiday",
            "linear",
            "random_walk",
            "sine",
        ]:
            self.assertIn(name, generators.get_all())
            self.assertTrue(hasattr(generators, name))
