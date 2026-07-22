import unittest

import numpy as np
import xarray as xr

from ontime.core.utils.restriction import Restriction


class TestRestriction(unittest.TestCase):
    def setUp(self):
        self.xa = xr.DataArray(np.array([1, 2, 3]))

    def test_check__restriction_satisfied__should_not_raise(self):
        restriction = Restriction("all positive", lambda xa: np.all(xa > 0))
        try:
            restriction.check(self.xa)
        except AssertionError:
            self.fail("check() raised AssertionError unexpectedly")

    def test_check__restriction_violated__should_raise_assertion_error_with_name(self):
        restriction = Restriction("all negative", lambda xa: np.all(xa < 0))
        with self.assertRaisesRegex(AssertionError, "all negative"):
            restriction.check(self.xa)
