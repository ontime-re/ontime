import unittest

import numpy as np
import pandas as pd

from ontime.core.processing.registry.mapper import Mapper
from ontime.core.time_series import TimeSeries


def make_ts(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return TimeSeries.from_times_and_values(index, np.array(values, dtype=float))


class TestMapper(unittest.TestCase):
    def test_process__doubling_function__should_apply_it_to_every_value(self):
        ts = make_ts([1, 2, 3, 4, 5])
        mapper = Mapper(fn=lambda x: x * 2)
        mapped = mapper.process(ts)
        np.testing.assert_array_equal(
            mapped.values().flatten(), np.array([2, 4, 6, 8, 10])
        )

    def test_inverse_process__invertible_mapper__should_recover_original_values(self):
        ts = make_ts([1, 2, 3, 4, 5])
        mapper = Mapper(fn=lambda x: x * 2, inverse_fn=lambda x: x / 2)
        mapped = mapper.process(ts)
        recovered = mapper.inverse_process(mapped)
        np.testing.assert_array_almost_equal(
            recovered.values().flatten(), ts.values().flatten()
        )

    def test_inverse_process__non_invertible_mapper__should_raise_not_implemented_error(
        self,
    ):
        ts = make_ts([1, 2, 3])
        mapper = Mapper(fn=lambda x: x * 2)
        mapped = mapper.process(ts)
        with self.assertRaises(NotImplementedError):
            mapper.inverse_process(mapped)
