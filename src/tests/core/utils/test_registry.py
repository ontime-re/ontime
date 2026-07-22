import unittest

from ontime.core.utils.registry import Registry


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()

    def test_constructor__creation__should_start_with_empty_registry(self):
        self.assertEqual(self.registry.registry, {})

    def test_register__new_item__should_store_it_under_name(self):
        self.registry.register("foo", 1)
        self.assertEqual(self.registry.registry["foo"], 1)

    def test_get__registered_item__should_return_it(self):
        self.registry.register("foo", 1)
        self.assertEqual(self.registry.get("foo"), 1)

    def test_get__missing_item__should_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.get("missing")

    def test_get_all__multiple_items__should_return_list_of_names(self):
        self.registry.register("foo", 1)
        self.registry.register("bar", 2)
        self.assertCountEqual(self.registry.get_all(), ["foo", "bar"])

    def test_update__existing_item__should_replace_value(self):
        self.registry.register("foo", 1)
        self.registry.update("foo", 2)
        self.assertEqual(self.registry.get("foo"), 2)

    def test_delete__existing_item__should_remove_it_from_registry(self):
        self.registry.register("foo", 1)
        self.registry.delete("foo")
        self.assertNotIn("foo", self.registry.registry)

    def test_delete__missing_item__should_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.delete("missing")
