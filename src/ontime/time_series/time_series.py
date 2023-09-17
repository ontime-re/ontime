from __future__ import annotations

from darts import TimeSeries as DartsTimeSeries
import pandas as pd
import xarray as xr

from typing import List


class TimeSeries(DartsTimeSeries):
    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)

    def split_by_period(self, period: str) -> List[TimeSeries]:
        """
        Split a time series given a period

        The period is defined with offset aliases from Pandas https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        :param period: period to split by
        :return: List of TimeSeries
        """
        df = self.pd_dataframe()
        splits_df = [g for n, g in df.groupby(pd.Grouper(freq=period))]
        splits_ts = list(map(self.from_dataframe, splits_df))
        return splits_ts

    @staticmethod
    def group_splits(split_ts: List[TimeSeries]) -> TimeSeries:
        """
        Group a list of time series into a single time series

        :param split_ts:  List of TimeSeries
        :return: TimeSeries
        """
        ts = split_ts[0]
        for ts_i in split_ts[1:]:
            ts = ts.append(ts_i)
        return ts

    @staticmethod
    def from_darts(ts):
        """
        Convert a Darts TimeSeries to an OnTime TimeSeries

        :param ts: Darts TimeSeries
        :return: OnTime TimeSeries
        """
        return TimeSeries(ts.data_array())
