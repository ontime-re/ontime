import unittest

from ontime.core.utils.utils import get_number_of_containing_entries


class TestGetNumberOfContainingEntries(unittest.TestCase):
    def test_get_number_of_containing_entries__equal_frequencies__should_return_one(
        self,
    ):
        self.assertEqual(get_number_of_containing_entries("D", "D"), 1)

    def test_get_number_of_containing_entries__freq2_is_double_freq1__should_return_two(
        self,
    ):
        self.assertEqual(get_number_of_containing_entries("D", "2D"), 2)

    def test_get_number_of_containing_entries__hourly_vs_daily__should_return_twenty_four(
        self,
    ):
        self.assertEqual(get_number_of_containing_entries("h", "D"), 24)

    def test_get_number_of_containing_entries__is_symmetric_regardless_of_argument_order(
        self,
    ):
        self.assertEqual(
            get_number_of_containing_entries("h", "D"),
            get_number_of_containing_entries("D", "h"),
        )

    def test_get_number_of_containing_entries__monthly_vs_daily__should_return_thirty(
        self,
    ):
        self.assertEqual(get_number_of_containing_entries("MS", "D"), 30)
