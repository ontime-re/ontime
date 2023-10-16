from typing import Generic, TypeVar, Optional, Union, List, Dict, Sequence, Callable, Type
import pandas as pd
from .time_series import TimeSeries
import xarray as xr
import numpy as np
from darts import TimeSeries as DartsTimeSeries

T = TypeVar('T')


class RestrictedTimeSeries(TimeSeries, Generic[T]):

    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)

    def check(self, xa: xr.DataArray) -> bool:
        raise NotImplementedError

    @classmethod
    def from_darts(cls, ts: DartsTimeSeries):
        """
        Convert a Darts TimeSeries to an OnTime TimeSeries

        :param ts: Darts TimeSeries
        :return: OnTime TimeSeries
        """
        cls.check(cls, ts.data_array())
        return cls(ts.data_array())

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
    ) -> Type[T]:
        cls.check(cls, xa)
        return super().from_xarray(xa, fill_missing_dates, freq, fillna_value)

    def append(self, other: T) -> T:
        """
        Append another T to this one

        :param other: T to append
        :return: T
        """

        return super().append(other)

    def append_values(self, values: np.ndarray) -> T:
        """
        Append values to the time series
        """
        raise NotImplementedError

    def concatenate(
            self,
            series: Sequence[T],
            axis: Union[str, int] = 0,
            ignore_time_axis: bool = False,
            ignore_static_covariates: bool = False,
            drop_hierarchy: bool = True,
    ):
        if series is T:
            return T(
                super().concatenate(series, axis, ignore_time_axis, ignore_static_covariates, drop_hierarchy))
        else:
            raise ValueError(f"Other's type must be {T.__name__}")

    def map(
            self,
            fn: Union[
                Callable[[np.number], np.number],
                Callable[[Union[pd.Timestamp, int], np.number], np.number],
            ],
    ):
        raise NotImplementedError

    def prepend(self, other: T) -> T:
        """
        Prepends (i.e. adds to the beginning) another series to this series along the time axis.

        :param other: The series to prepend.
        :return: The prepended series.
        """
        if other is T:
            return T(super().prepend(other))
        else:
            raise ValueError(f"Other's type must be {T.__name__}")

    def prepend_values(self, values: np.ndarray) -> T:
        """
        Prepends (i.e. adds to the beginning) values to this series along the time axis.

        :param values: The values to prepend.
        :return: The prepended series.
        """
        if isinstance(values, np.ndarray):
            ts = T(super().prepend_values(values))
            if ts.check(values):
                return ts
            else:
                raise ValueError("Values must be binary")
        else:
            raise ValueError(f"Other's type must be {T.__name__}")

    def rescale_with_value(self, value_at_first_step: float) -> 'TimeSeries':
        """
        Rescales the time series so that the first value is equal to the given value.

        :param value_at_first_step: The value at the first step of the rescaled time series.
        :return: The rescaled time series. **It changes the type of the ProbabilisticTimeSeries to TimeSeries.**
        TODO Check if this action make sense
        """
        return TimeSeries(super().rescale_with_value(value_at_first_step))

    def stack(self, other: "TimeSeries") -> "TimeSeries":
        """
        Stacks this time series with another one, along the time axis.

        :param other: The time series to stack with.
        :return: The stacked time series.  **It changes the type of the ProbabilisticTimeSeries to TimeSeries.**
        TODO Check if this action make sense
        """
        return TimeSeries(super().stack(other))

    def sum(self, axis: int = 2) -> 'TimeSeries':
        """
        Sums the values along the given axis.

        :param axis: The axis along which to sum.
        :return: The summed time series.  **It changes the type of the T to TimeSeries.**
        TODO Check if this action make sense
        """
        return TimeSeries(super().sum(axis))

    def window_transform(
            self,
            transforms: Union[Dict, Sequence[Dict]],
            treat_na: Optional[Union[str, Union[int, float]]] = None,
            forecasting_safe: Optional[bool] = True,
            keep_non_transformed: Optional[bool] = False,
            include_current: Optional[bool] = True,
    ):
        raise NotImplementedError

    def with_values(self, values: np.ndarray) -> T:
        """
        Returns a copy of the time series with the given values.

        :param values: The values to set.
        :return: The time series with the given values.
        """
        if isinstance(values, np.ndarray):
            ts = T(super().with_values(values))
            if ts.check(values):
                return ts
            else:
                raise ValueError("Values must be binary")
        else:
            raise ValueError(f"Other's type must be {T.__name__}")
