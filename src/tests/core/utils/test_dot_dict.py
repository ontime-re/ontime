import unittest

from ontime.core.utils.dot_dict import DotDict


class TestDotDict(unittest.TestCase):
    def test_getattr__existing_key__should_return_value(self):
        d = DotDict({"a": 1})
        self.assertEqual(d.a, 1)

    def test_getattr__missing_key__should_raise_attribute_error(self):
        d = DotDict()
        with self.assertRaises(AttributeError):
            _ = d.missing

    def test_setattr__new_attribute__should_store_as_dict_key(self):
        d = DotDict()
        d.a = 42
        self.assertEqual(d["a"], 42)
        self.assertEqual(d.a, 42)

    def test_delattr__existing_key__should_remove_key(self):
        d = DotDict({"a": 1})
        del d.a
        self.assertNotIn("a", d)

    def test_delattr__missing_key__should_raise_attribute_error(self):
        d = DotDict()
        with self.assertRaises(AttributeError):
            del d.missing
