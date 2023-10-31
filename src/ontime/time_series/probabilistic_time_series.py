from .resticted_time_series import RestrictedTimeSeries
import xarray as xr
import numpy as np
import pandas as pd


class ProbabilisticTimeSeries(RestrictedTimeSeries["ProbabilisticTimeSeries"]):
    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)
        if not self.check(xa):
            raise ValueError("Input DataArray contains values outside the range [0, 1]")

    def check(self, xa: xr.DataArray):
        """
        Check if all values in the xarray DataArray are between 0 and 1.

        Parameters:
        data_array (xr.DataArray): The xarray DataArray to be checked.

        Returns:
        bool: True if all values are between 0 and 1, False otherwise.
        """
        if isinstance(xa, xr.DataArray):
            # Check if all values in the DataArray are within the range [0, 1]
            return np.all((xa >= 0) & (xa <= 1))
        else:
            raise ValueError("Input is not an xarray DataArray")
