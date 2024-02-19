from __future__ import annotations
from typing import List
from darts import TimeSeries as DartsTimeSeries
import pandas as pd
import xarray as xr


class TimeSeries(DartsTimeSeries):
    """
    Main class to handle time series
    This is a wrapper around Darts TimeSeries, functions are added to handle various operations
    """

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
    def from_darts(ts: DartsTimeSeries) -> TimeSeries:
        """
        Convert a Darts TimeSeries to an OnTime TimeSeries

        :param ts: Darts TimeSeries
        :return: OnTime TimeSeries
        """
        return TimeSeries(ts.data_array())

    @staticmethod
    def from_pandas(df: pd.DataFrame, freq=None) -> TimeSeries:
        """
        Convert a pandas DataFrame to an OnTime TimeSeries using Darts
        Assumes the index is compliant with TimeEval's canonical format.

        :param df: pandas dataFrame
        :return: OnTime TimeSeries
        """
        ts = DartsTimeSeries.from_dataframe(df, fill_missing_dates=True, freq=freq)
        return TimeSeries.from_darts(ts)

    @staticmethod
    def from_csv(file: str, index_col=None) -> TimeSeries:
        """
        Reads a csv file and converts it to an OnTime TimeSeries using Darts and pandas.
        Assumes the csv file is compliant with TimeEval's canonical format.

        :param file: location of the dataset csv file
        :param index_col: the name of the column to be used for the index
        :return: OnTime TimeSeries
        """
        df = pd.read_csv(file)
        if index_col is not None:
            df.index = df[index_col].astype("datetime64[ns]")
            del df[index_col]
        return TimeSeries.from_pandas(df)

    @staticmethod
    def from_data(data, index=None, columns=None) -> TimeSeries:
        """
        Converts data to a TimeSeries. Takes data as a dict, ndarray, Iterable or pandas DataFrame.
        Assumes the index is compliant with TimeEval's canonical format.

        :param data: a dict, ndarray, Iterable or pandas DataFrame
        :param index: the array-like series to be used as index (will default to a range if none is specified)
        :param columns: will be used for column names if there are none in the data parameter (will default to a range if nothing is specified)
        :return: OnTime TimeSeries
        """
        df = pd.DataFrame(data, index, columns, copy=True)
        return TimeSeries.from_pandas(df)
