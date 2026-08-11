import numpy as np

from .restricted_time_series import RestrictedTimeSeries
from ..utils.restriction import Restriction


class UnitTimeSeries(RestrictedTimeSeries):
    """
    A time series with restrictions on the data so that all values are between 0 and 1.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the UnitTimeSeries.

        :param args: positional arguments passed to the Darts TimeSeries constructor.
        :param kwargs: keyword arguments passed to the Darts TimeSeries constructor.
        """
        super().__init__(*args, **kwargs)
        self.restriction = Restriction("Unit Restriction", self.unit_restriction)
        self.add_restriction(self.restriction)

    @staticmethod
    def unit_restriction(values: np.ndarray) -> bool:
        """
        Check if all values are between 0 and 1.

        :param values: The values to check.
        :return: bool
        """
        return np.all((values >= 0) & (values <= 1))
