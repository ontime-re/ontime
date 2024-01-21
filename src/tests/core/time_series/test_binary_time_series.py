import unittest

import pandas as pd
import numpy as np

from ontime.core.time_series.binary_time_series import BinaryTimeSeries

class TestBinaryTimeSeries(unittest.TestCase):

    def setUp(self):
        # Create a binary dictionary with index as datetime and values as 0 or 1
        keys = pd.date_range(start='01-01-2024', end='01-02-2024', freq='1H')
        values = np.random.randint(0, 2, size=len(keys))
        my_dict = {
            "times": keys,
            "values": values
        }
        # Create a pandas dataframe from the dictionary
        df = pd.DataFrame.from_dict(my_dict)
        df.set_index('times', inplace=True)
        self.df = df

    def test_constructor__creation_from_dataframe__should_create_object_with_correct_data(self):
        bts = BinaryTimeSeries.from_dataframe(self.df)
        self.assertIsInstance(bts, BinaryTimeSeries)
        for i in bts.values():
            is_binary = i[0] == 0 or i[0] == 1
            self.assertTrue(is_binary, "value should be 0 or 1")