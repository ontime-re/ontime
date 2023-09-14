from darts.dataprocessing.transformers.mappers import Mapper as DartsMapper
from darts.dataprocessing.transformers.mappers import (
    InvertibleMapper as DartsInvertibleMapper,
)

from ...abstract import AbstractBaseProcessor
from ...time_series import TimeSeries


class Mapper(AbstractBaseProcessor):
    def __init__(self, func, inverse_func=None, name="Map", n_jobs=1, verbose=False):
        super().__init__()
        self.func = func

        if inverse_func is None:
            self._processor = DartsMapper(
                func, name=name, n_jobs=n_jobs, verbose=verbose
            )
        else:
            self._processor = DartsInvertibleMapper(
                func, inverse_func, name=name, n_jobs=n_jobs, verbose=verbose
            )

    def process(self, ts) -> TimeSeries:
        return self._processor.transform(ts)

    def inverse_process(self, ts) -> TimeSeries:
        if isinstance(self._processor, DartsInvertibleMapper):
            return self._processor.inverse_transform(ts)
        else:
            raise NotImplementedError(
                "This mapper is not invertible, add a inverse_func to the constructor"
            )
