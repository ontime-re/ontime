from darts.dataprocessing.transformers.missing_values_filler import (
    MissingValuesFiller as DartsMissingValuesFiller,
)

from ...abstract import AbstractBaseProcessor
from ...time_series import TimeSeries


class Filler(AbstractBaseProcessor):
    """Wrapper around Darts MissingValuesFiller.
    https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.missing_values_filler.html
    """

    def __init__(self, fill="auto", name="Fill", n_jobs=1, verbose=False):
        """Data transformer to fill missing values from a (sequence of) deterministic ``TimeSeries``.

        :param fill: Union[str, Union[int, float]]
            The value used to replace the missing values.
            If set to 'auto', will auto-fill missing values using the :func:`pd.Dataframe.interpolate()` method.
        :param name: str
            A specific name for the transformer
        :param n_jobs: int
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
            Note: for a small amount of data, the parallelisation overhead could end up increasing the total
            required amount of time.
        :param verbose: bool
            Optionally, whether to print operations progress
        """
        super().__init__()
        self._processor = DartsMissingValuesFiller(
            fill=fill, name=name, n_jobs=n_jobs, verbose=verbose
        )

    def process(self, ts) -> TimeSeries:
        """Process the time series.
        :param ts: TimeSeries
        :return: TimeSeries
        """
        ts_processed = self._processor.transform(ts)
        return TimeSeries.from_darts(ts_processed)
