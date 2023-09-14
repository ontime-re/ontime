import xarray as xr
from darts import TimeSeries as DartsTimeSeries


class TimeSeries(DartsTimeSeries):
    def __init__(self, xa: xr.DataArray):
        super().__init__(xa)
