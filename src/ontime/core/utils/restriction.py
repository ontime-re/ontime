from typing import Callable

import xarray as xr


class Restriction:
    def __init__(self, name: str, restriction: Callable):
        self.name = name
        self.restriction = restriction

    def check(self, xa: xr.DataArray) -> None:
        assert self.restriction(xa), f"Restriction {self.name} failed"
