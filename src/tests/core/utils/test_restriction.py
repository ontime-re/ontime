import unittest

import numpy as np

from ontime.core.utils.restriction import Restriction


class TestRestriction(unittest.TestCase):
    def setUp(self):
        self.values = np.array([1.0, 2.0, 3.0])

    def test_check__restriction_satisfied__should_not_raise(self):
        restriction = Restriction("all positive", lambda values: np.all(values > 0))
        try:
            restriction.check(self.values)
        except AssertionError:
            self.fail("check() raised AssertionError unexpectedly")

    def test_check__restriction_violated__should_raise_assertion_error_with_name(self):
        restriction = Restriction("all negative", lambda values: np.all(values < 0))
        with self.assertRaisesRegex(AssertionError, "all negative"):
            restriction.check(self.values)
