from darts.dataprocessing.transformers.mappers import Mapper as DartsMapper
from darts.dataprocessing.transformers.mappers import (
    InvertibleMapper as DartsInvertibleMapper,
)

from ...abstract import AbstractBaseProcessor
from ...time_series import TimeSeries


class Mapper(AbstractBaseProcessor):
    """
    Wrapper around Darts Mapper
    https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.mappers.html
    """

    def __init__(self, fn, inverse_fn=None, name="Map", n_jobs=1, verbose=False):
        """Data transformer to apply a custom function and its inverse to a ``TimeSeries``
        (similar to calling :func:`TimeSeries.map()` on each series).

        :param fn:
            Either a function which takes a value and returns a value ie. `f(x) = y`
            Or a function which takes a value and its timestamp and returns a value ie. `f(timestamp, x) = y`.
        :param inverse_fn:
            Similarly to `fn`, either a function which takes a value and returns a value ie. `f(x) = y`
            Or a function which takes a value and its timestamp and returns a value ie. `f(timestamp, x) = y`.
            `inverse_fn` should be such that ``inverse_fn(fn(x)) == x``.        :param name:
        :param name:
            A specific name for the transformer.
        :param n_jobs:
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input to a method, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        :param verbose:
            Optionally, whether to print operations progress
        """
        super().__init__()
        self.func = fn

        if inverse_fn is None:
            self._processor = DartsMapper(fn, name=name, n_jobs=n_jobs, verbose=verbose)
        else:
            self._processor = DartsInvertibleMapper(
                fn, inverse_fn, name=name, n_jobs=n_jobs, verbose=verbose
            )

    def process(self, ts) -> TimeSeries:
        """Process the time series.

        :param ts: TimeSeries
        :return: TimeSeries
        """
        return self._processor.transform(ts)

    def inverse_process(self, ts) -> TimeSeries:
        """Inverse process the time series.

        :param ts: TimeSeries
        :return: TimeSeries
        """
        if isinstance(self._processor, DartsInvertibleMapper):
            return self._processor.inverse_transform(ts)
        else:
            raise NotImplementedError(
                "This mapper is not invertible, add a inverse_func to the constructor"
            )
