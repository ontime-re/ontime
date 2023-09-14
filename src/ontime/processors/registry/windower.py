from typing import Union, List, Optional

from darts.dataprocessing.transformers.window_transformer import (
    WindowTransformer as DartsWindowTransformer,
)

from ...abstract import AbstractBaseProcessor
from ...time_series import TimeSeries


class Windower(AbstractBaseProcessor):
    """
    Apply windowing function on a time series.
    https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.window_transformer.html#window-transformer
    """

    def __init__(
        self,
        transforms: Union[dict, List[dict]],
        treat_na: Optional[Union[str, Union[int, float]]] = None,
        forecasting_safe: Optional[bool] = True,
        keep_non_transformed: Optional[bool] = False,
        include_current: Optional[bool] = True,
        name: str = "Windower",
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        self._processor = DartsWindowTransformer(
            transforms,
            treat_na=treat_na,
            forecasting_safe=forecasting_safe,
            keep_non_transformed=keep_non_transformed,
            include_current=include_current,
            name=name,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def process(self, ts: TimeSeries) -> TimeSeries:
        return self._processor.transform(ts)
