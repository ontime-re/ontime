from typing import Optional, Union, List, Dict, Sequence, Callable

import pandas as pd
import xarray as xr
import numpy as np

from .time_series import TimeSeries
from .probabilistic_time_series import ProbabilisticTimeSeries


class BinaryTimeSeries(ProbabilisticTimeSeries):

    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)
        if not BinaryTimeSeries.is_binary(xa):
            raise ValueError("Input DataArray contains values outside of {0; 1}")

    @staticmethod
    def is_binary(xa: xr.DataArray):
        """
        Check if all values in the xarray DataArray are between 0 and 1.

        Parameters:
        data_array (xr.DataArray): The xarray DataArray to be checked.

        Returns:
        bool: True if all values are between 0 and 1, False otherwise.
        """
        if isinstance(xa, xr.DataArray):
            # Check if all values in the DataArray are within {0; 1}
            return np.all((xa == 0) & (xa == 1))
        else:
            raise ValueError("Input is not an xarray DataArray")

    @staticmethod
    def is_binary(values: np.ndarray):
        if isinstance(values, np.ndarray):
            return np.all((values == 0) & (values == 1))
        else:
            raise ValueError("Input is not an np.ndarray")

    @staticmethod
    def from_darts(ts):
        """
        Convert a Darts TimeSeries to an OnTime TimeSeries

        :param ts: Darts TimeSeries
        :return: OnTime TimeSeries
        """
        BinaryTimeSeries.is_binary(ts.data_array())
        return BinaryTimeSeries(ts.data_array())

    @classmethod
    def from_dataframe(
            cls,
            df: pd.DataFrame,
            time_col: Optional[str] = None,
            value_cols: Optional[Union[List[str], str]] = None,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[Union[str, int]] = None,
            fillna_value: Optional[float] = None,
            static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
            hierarchy: Optional[Dict] = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_group_dataframe(
            cls,
            df: pd.DataFrame,
            group_cols: Union[List[str], str],
            time_col: Optional[str] = None,
            value_cols: Optional[Union[List[str], str]] = None,
            static_cols: Optional[Union[List[str], str]] = None,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[Union[str, int]] = None,
            fillna_value: Optional[float] = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_series(
            cls,
            pd_series: pd.Series,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[Union[str, int]] = None,
            fillna_value: Optional[float] = None,
            static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_values(
            cls,
            values: np.ndarray,
            columns: Optional[pd._typing.Axes] = None,
            fillna_value: Optional[float] = None,
            static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
            hierarchy: Optional[Dict] = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_pickle(cls, path: str):
        raise NotImplementedError

    @classmethod
    def from_times_and_values(
            cls,
            times: Union[pd.DatetimeIndex, pd.RangeIndex, pd.Index],
            values: np.ndarray,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[Union[str, int]] = None,
            columns: Optional[pd._typing.Axes] = None,
            fillna_value: Optional[float] = None,
            static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
            hierarchy: Optional[Dict] = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_csv(
            cls,
            filepath_or_buffer,
            time_col: Optional[str] = None,
            value_cols: Optional[Union[List[str], str]] = None,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[Union[str, int]] = None,
            fillna_value: Optional[float] = None,
            static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
            hierarchy: Optional[Dict] = None,
            **kwargs,
    ):
        raise NotImplementedError

    @classmethod
    def from_json(
            cls,
            json_str: str,
            static_covariates: Optional[Union[pd.Series, pd.DataFrame]] = None,
            hierarchy: Optional[Dict] = None,
    ):
        raise NotImplementedError

    @classmethod
    def from_xarray(
            cls,
            xa: xr.DataArray,
            fill_missing_dates: Optional[bool] = False,
            freq: Optional[Union[str, int]] = None,
            fillna_value: Optional[float] = None,
    ):
        raise NotImplementedError

    def append(self, other: 'BinaryTimeSeries') -> 'BinaryTimeSeries':
        """
        Append another BinaryTimeSeries to this one

        :param other: BinaryTimeSeries to append
        :return: BinaryTimeSeries
        """
        if isinstance(other, BinaryTimeSeries):
            return BinaryTimeSeries(super().append(other))
        else:
            raise ValueError("Other's type must be BinaryTimeSeries")

    def append_values(self, values: np.ndarray) -> 'BinaryTimeSeries':
        """
        Append values to the time series

        :param values: np.ndarray
        :return: BinaryTimeSeries
        """
        # check that all value of values are binary
        if isinstance(values, np.ndarray):
            if BinaryTimeSeries.is_binary(values):
                return BinaryTimeSeries(super().append_values(values))
            else:
                raise ValueError("Values must be binary")
        else:
            raise ValueError("Other's type must be BinaryTimeSeries")

    def concatenate(
            self,
            series: Sequence["BinaryTimeSeries"],
            axis: Union[str, int] = 0,
            ignore_time_axis: bool = False,
            ignore_static_covariates: bool = False,
            drop_hierarchy: bool = True,
    ):
        if isinstance(series, BinaryTimeSeries):
            return BinaryTimeSeries(
                super().concatenate(series, axis, ignore_time_axis, ignore_static_covariates, drop_hierarchy))
        else:
            raise ValueError("Other's type must be BinaryTimeSeries")

    def map(
            self,
            fn: Union[
                Callable[[np.number], np.number],
                Callable[[Union[pd.Timestamp, int], np.number], np.number],
            ],
    ) :
        raise NotImplementedError

    def prepend(self, other: "BinaryTimeSeries") -> "BinaryTimeSeries":
        """
        Prepends (i.e. adds to the beginning) another series to this series along the time axis.

        :param other: The series to prepend.
        :return: The prepended series.
        """
        if isinstance(other, BinaryTimeSeries):
            return BinaryTimeSeries(super().prepend(other))
        else:
            raise ValueError("Other's type must be BinaryTimeSeries")

    def prepend_values(self, values: np.ndarray) -> 'BinaryTimeSeries':
        """
        Prepends (i.e. adds to the beginning) values to this series along the time axis.

        :param values: The values to prepend.
        :return: The prepended series.
        """
        if isinstance(values, np.ndarray):
            if BinaryTimeSeries.is_binary(values):
                return BinaryTimeSeries(super().prepend_values(values))
            else:
                raise ValueError("Values must be binary")
        else:
            raise ValueError("Other's type must be BinaryTimeSeries")

    def rescale_with_value(self, value_at_first_step: float) -> 'TimeSeries':
        """
        Rescales the time series so that the first value is equal to the given value.

        :param value_at_first_step: The value at the first step of the rescaled time series.
        :return: The rescaled time series. **It changes the type of the BinaryTimeSeries to TimeSeries.**
        TODO Check if this action make sense for BinaryTimeSeries
        """
        return TimeSeries(super().rescale_with_value(value_at_first_step))


    def stack(self, other: "TimeSeries") -> "TimeSeries":
        """
        Stacks this time series with another one, along the time axis.

        :param other: The time series to stack with.
        :return: The stacked time series.  **It changes the type of the BinaryTimeSeries to TimeSeries.**
        TODO Check if this action make sense for BinaryTimeSeries
        """
        return TimeSeries(super().stack(other))

    def sum(self, axis: int = 2) -> 'TimeSeries':
        """
        Sums the values along the given axis.

        :param axis: The axis along which to sum.
        :return: The summed time series.  **It changes the type of the BinaryTimeSeries to TimeSeries.**
        TODO Check if this action make sense for BinaryTimeSeries
        """
        return TimeSeries(super().sum(axis))



    def window_transform(
        self,
        transforms: Union[Dict, Sequence[Dict]],
        treat_na: Optional[Union[str, Union[int, float]]] = None,
        forecasting_safe: Optional[bool] = True,
        keep_non_transformed: Optional[bool] = False,
        include_current: Optional[bool] = True,
    ) :
        raise NotImplementedError

    def with_values(self, values: np.ndarray) -> 'BinaryTimeSeries':
        """
        Returns a copy of the time series with the given values.

        :param values: The values to set.
        :return: The time series with the given values.
        """
        if isinstance(values, np.ndarray):
            if BinaryTimeSeries.is_binary(values):
                return BinaryTimeSeries(super().with_values(values))
            else:
                raise ValueError("Values must be binary")
        else:
            raise ValueError("Other's type must be BinaryTimeSeries")
