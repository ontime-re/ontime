import unittest

import altair as alt
import pandas as pd
import torch

from ontime.core.time_series import TimeSeries


class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-01", periods=4, freq="D")
        self.df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=self.index)
        self.df.index.name = "time"
        self.multi_df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]},
            index=self.index,
        )
        self.multi_df.index.name = "time"

    def test_from_pandas__dataframe__should_create_time_series_with_same_values(self):
        ts = TimeSeries.from_pandas(self.df)
        self.assertIsInstance(ts, TimeSeries)
        self.assertListEqual(list(ts.values().flatten()), [1.0, 2.0, 3.0, 4.0])

    def test_from_data__dict_with_index__should_create_time_series(self):
        ts = TimeSeries.from_data({"x": [1, 2, 3]}, index=self.index[:3])
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(list(ts.components), ["x"])
        self.assertListEqual(list(ts.values().flatten()), [1.0, 2.0, 3.0])

    def test_rename__mapping_of_column_names__should_rename_components(self):
        ts = TimeSeries.from_pandas(self.df)
        renamed = ts.rename({"a": "b"})
        self.assertListEqual(list(renamed.components), ["b"])

    def test_to_tensor__time_series__should_return_torch_tensor_with_same_values(self):
        ts = TimeSeries.from_pandas(self.df)
        tensor = ts.to_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tuple(tensor.shape), (4, 1))

    def test_split_by_period_and_group_splits__round_trip__should_recover_original_series(
        self,
    ):
        ts = TimeSeries.from_pandas(self.df)
        splits = ts.split_by_period("2D")
        self.assertEqual(len(splits), 2)
        grouped = TimeSeries.group_splits(splits)
        self.assertListEqual(
            list(grouped.values().flatten()), list(ts.values().flatten())
        )

    def test_from_darts__darts_time_series__should_convert_to_ontime_time_series(self):
        ts = TimeSeries.from_pandas(self.df)
        darts_ts = ts  # TimeSeries subclasses DartsTimeSeries
        converted = TimeSeries.from_darts(darts_ts)
        self.assertIsInstance(converted, TimeSeries)
        self.assertListEqual(
            list(converted.values().flatten()), list(ts.values().flatten())
        )

    def test_plot__default_multivariate_series__should_return_layer_chart(self):
        ts = TimeSeries.from_pandas(self.multi_df)

        chart = ts.plot()

        self.assertIsInstance(chart, alt.LayerChart)

    def test_plot__subplots_true__should_return_facet_chart_per_variable(self):
        ts = TimeSeries.from_pandas(self.multi_df)

        chart = ts.plot(subplots=True)

        self.assertIsInstance(chart, alt.FacetChart)
        self.assertEqual(chart.facet.row.shorthand, "variable:N")
        self.assertEqual(chart.resolve.scale.y, "independent")
