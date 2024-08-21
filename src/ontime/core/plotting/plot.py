from typing import Callable

from pandas import DataFrame
import altair as alt

from ..time_series import TimeSeries


class Plot:
    def __init__(self, ts: TimeSeries = None, **kwargs):
        super().__init__(**kwargs)
        Plot.config()
        self.ts = ts
        self.layers = []

    def add(self, mark: Callable, ts: TimeSeries = None, **kwargs):
        """
        Add a mark to the plot

        :param mark: The mark function to use
        :param ts: the series to plot
        :param kwargs: the arguments to pass to the mark function
        :return: Plot
        """
        ts = self.ts if ts is None else ts
        self.layers.append(mark(ts, **kwargs))
        return self

    def properties(self, **kwargs):
        """
        Set properties of the last layer (and the plot)

        :param kwargs: The properties to set
        :return: Plot
        """
        self.layers[-1] = self.layers[-1].properties(**kwargs)
        return self

    def show(self):
        """
        Show the plot

        :return: Altair LayerChart
        """
        return alt.layer(*self.layers)

    @staticmethod
    def melt(ts: TimeSeries) -> DataFrame:
        """
        Melt a TimeSeries into a DataFrame

        :param ts: TimeSeries
        :return: DataFrame
        """
        df = ts.pd_dataframe()
        df = df.reset_index()
        df = df.melt("time", var_name="variable", value_name="value")
        return df

    @staticmethod
    def config():
        """
        Configure the Plot

        :return: None
        """
        alt.data_transformers.enable("vegafusion")
