from __future__ import annotations
from typing import List

from darts import TimeSeries as DartsTimeSeries
import pandas as pd
import xarray as xr
from torch import Tensor


class TimeSeries(DartsTimeSeries):
    """
    Main class to handle time series
    This is a wrapper around Darts TimeSeries, functions are added to handle various operations
    """

    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)

    def plot(self, width: int = 400, height: int = 200, **kwargs):
        """
        Plot the TimeSeries

        :param width: width of the plot
        :param height: height of the plot
        :param kwargs: additional arguments to pass to the line mark
        :return: Altair LayerChart
        """
        from ..plotting.plot import Plot
        from ..plotting._marks.line import line

        return (
            Plot(self).add(line, **kwargs).properties(width=width, height=height).show()
        )

    def rename(self, columns: dict):
        """
        Rename the columns of the TimeSeries

        :param columns: dict with {"old_name": "new_name"}
        """
        col_names = list(columns.keys())
        col_names_new = list(columns.values())
        return self.with_columns_renamed(
            col_names=col_names, col_names_new=col_names_new
        )

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
        Convert a pandas DataFrame to an onTime TimeSeries using Darts
        Assumes the index is compliant with TimeEval's canonical format.

        :param df: pandas dataFrame
        :param freq: frequency of the time series
        :return: onTime TimeSeries
        """
        ts = DartsTimeSeries.from_dataframe(df, fill_missing_dates=True, freq=freq)
        return TimeSeries.from_darts(ts)

    @staticmethod
    def from_csv(file: str, index_col=None) -> TimeSeries:
        """
        Reads a csv file and converts it to an onTime TimeSeries using Darts and pandas.
        Assumes the csv file is compliant with TimeEval's canonical format.

        :param file: location of the dataset csv file
        :param index_col: the name of the column to be used for the index
        :return: onTime TimeSeries
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

    def to_tensor(self):
        """
        Convert the TimeSeries to a tensor
        """
        return Tensor(self.values())
