import unittest

from ontime.core.utils.dynamic_class import DynamicClass


class TestDynamicClass(unittest.TestCase):
    def setUp(self):
        self.dynamic_class = DynamicClass()

    def test_load__new_class__should_register_and_set_attribute(self):
        self.dynamic_class.load("my_int", int)
        self.assertEqual(self.dynamic_class.get("my_int"), int)
        self.assertIs(self.dynamic_class.my_int, int)

    def test_load_registry__pre_populated_registry__should_set_all_attributes(self):
        self.dynamic_class.register("my_str", str)
        self.dynamic_class.register("my_float", float)
        self.dynamic_class.load_registry()
        self.assertIs(self.dynamic_class.my_str, str)
        self.assertIs(self.dynamic_class.my_float, float)
