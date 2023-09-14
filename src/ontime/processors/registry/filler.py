from darts.dataprocessing.transformers.missing_values_filler import (
    MissingValuesFiller as DartsMissingValuesFiller,
)

from ...abstract import AbstractBaseProcessor
from ...time_series import TimeSeries


class Filler(AbstractBaseProcessor):
    """
    Fill missing values in a time series.
    https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.missing_values_filler.html
    """

    def __init__(self, fill="auto", name="Fill", n_jobs=1, verbose=False):
        super().__init__()
        self._processor = DartsMissingValuesFiller(
            fill=fill, name=name, n_jobs=n_jobs, verbose=verbose
        )

    def process(self, ts) -> TimeSeries:
        return self._processor.transform(ts)
