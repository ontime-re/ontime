import xarray as xr

from .time_series import TimeSeries
from ..utils.restriction import Restriction


class RestrictedTimeSeries(TimeSeries):
    """
    A time series with restrictions on the data.
    """

    def __init__(self, xa: xr.DataArray, restrictions=None):
        """
        Initialize the RestrictedTimeSeries.

        :param xa: Xarray DataArray to use.
        :param restrictions: List of Restriction objects to apply.
        """
        super().__init__(xa)

        self.restrictions = []
        if restrictions is not None:
            for r in restrictions:
                assert isinstance(
                    r, Restriction
                ), "All restrictions must be of type Restriction"
                self.add_restriction(r)

    def check(self) -> bool:
        """
        Check the restrictions on the time series.

        :return: None
        :raises: AssertionError
        """
        for restriction in self.restrictions:
            restriction.check(self.data_array())
        return True

    def add_restriction(self, restriction: Restriction) -> None:
        """
        Add a restriction to the time series.

        :param restriction: Restriction to add.
        :return: None
        """
        self.restrictions.append(restriction)
        self.check()
