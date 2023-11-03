from typing import Union
from datetime import timedelta

import pandas as pd
from pandas._libs.tslibs import BaseOffset
from pandas.core.indexers.objects import BaseIndexer
import numpy as np

from ...time_series import TimeSeries


class Correlation:
    """Correlation class handles correlation computation in a TimeSeries"""

    @staticmethod
    def process(
        ts: TimeSeries, window: Union[int, timedelta, str, BaseOffset, BaseIndexer]
    ) -> TimeSeries:
        """Compute correlations for a TimeSeries

        :param ts: TimeSeries
        :param window: int, timedelta, str, offset, or BaseIndexer subclass
            Size of the moving window as in https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html#pandas-dataframe-rolling
        :return: TimeSeries
            Each correlation is a component of the TimeSeries with a name such as 'var_a_var_b'
        """
        df = ts.pd_dataframe()
        df = Correlation.compute_correlations(df, window)
        df = Correlation.pivot(df)
        df.columns.name = None  # Otherwise, the column name is 'pair' and from_dataframe() fails in the next line
        return TimeSeries.from_dataframe(df)

    @staticmethod
    def compute_correlations(
        df: pd.DataFrame, window: Union[int, timedelta, str, BaseOffset, BaseIndexer]
    ) -> pd.DataFrame:
        """
        Compute correlations for a DataFrame

        :param df: DataFrame
        :param window: int, timedelta, str, offset, or BaseIndexer subclass
            Size of the moving window as in https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html#pandas-dataframe-rolling
        :return: DataFrame (in long format)
        """
        df = df.rolling(window=window).corr()

        # Step 1: Get unique time indices (level 0 of your MultiIndex)
        unique_times = df.index.get_level_values(0).unique()

        # Prepare a container for non-redundant items
        non_redundant_items = []

        for time in unique_times:
            # Step 2: Get the slice of the dataframe corresponding to the current time
            current_df = df.xs(time, level=0)

            # Step 3: Extract upper triangle of the correlation matrix without the diagonal
            # Since indices and columns are the same, we can assume it's a square matrix
            upper_triangle_indices = np.triu_indices(
                n=current_df.shape[0], k=1
            )  # k=1 excludes the main diagonal
            upper_triangle_values = current_df.values[upper_triangle_indices]

            # Step 4: Store non-redundant items with their respective labels and time
            # We iterate directly over the upper_triangle_indices, which are pairs of (row, col) positions in the matrix
            for (row, col), value in zip(
                zip(*upper_triangle_indices), upper_triangle_values
            ):
                non_redundant_items.append(
                    {
                        "time": time,
                        "var_a": current_df.index[row],
                        "var_b": current_df.columns[col],
                        "correlation": value,
                    }
                )

        # Step 5: Convert the list of non-redundant items to a new DataFrame
        return pd.DataFrame(non_redundant_items)

    @staticmethod
    def pivot(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot a DataFrame so that each pair of variables becomes a column

        :param df: DataFrame
        :return: DataFrame (in wide format)
        """

        # Step 1: Set 'time' as the index
        df.set_index("time", inplace=True)

        # Step 2: Create a unique identifier for each (row_index, col_index) pair
        # This will become the column names in the reshaped DataFrame
        df["pair"] = df["var_a"] + "_" + df["var_b"]

        # Step 3: Pivot the table so that each 'pair' becomes a column
        # The values in the table will be the 'correlation' values
        return df.pivot(columns="pair", values="correlation")
