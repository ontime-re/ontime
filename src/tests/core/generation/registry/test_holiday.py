import unittest

import pandas as pd

from ontime.core.generation.registry.holiday import Holiday
from ontime.core.time_series import TimeSeries


class TestHoliday(unittest.TestCase):
    def test_generate__known_us_holiday__should_flag_christmas_day(self):
        time_index = pd.date_range("2024-12-24", periods=5, freq="D")
        ts = Holiday().generate(time_index, country_code="US")
        self.assertIsInstance(ts, TimeSeries)
        values = ts.values().flatten()
        christmas_position = list(time_index).index(pd.Timestamp("2024-12-25"))
        self.assertEqual(values[christmas_position], 1.0)

    def test_generate__non_holiday_day__should_not_be_flagged(self):
        time_index = pd.date_range("2024-12-24", periods=5, freq="D")
        ts = Holiday().generate(time_index, country_code="US")
        values = ts.values().flatten()
        non_holiday_position = list(time_index).index(pd.Timestamp("2024-12-24"))
        self.assertEqual(values[non_holiday_position], 0.0)
