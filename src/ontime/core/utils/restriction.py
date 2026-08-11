from typing import Callable

import numpy as np


class Restriction:
    def __init__(self, name: str, restriction: Callable):
        self.name = name
        self.restriction = restriction

    def check(self, values: np.ndarray) -> None:
        assert self.restriction(values), f"Restriction {self.name} failed"
