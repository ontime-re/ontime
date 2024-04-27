import xarray as xr
import numpy as np

from .restricted_time_series import RestrictedTimeSeries
from ..utils.restriction import Restriction


class BinaryTimeSeries(RestrictedTimeSeries):
    """
    A time series with restrictions on the data so that all values are either 0 or 1.
    """

    def __init__(self, xa: xr.DataArray):
        """
        Initialize the BinaryTimeSeries.

        :param xa: Xarray DataArray to use.
        """
        super().__init__(xa)
        self.restriction = Restriction("Binary Restriction", self.binary_restriction)
        self.add_restriction(self.restriction)

    @staticmethod
    def binary_restriction(xa: xr.DataArray) -> bool:
        """
        Check if all values in the data array are either 0 or 1.

        :param xa: The Xarray DataArray to check.
        :return: bool
        """
        return np.all((xa == 0) | (xa == 1))
