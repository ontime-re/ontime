from typing import Union, List, Optional

from darts.dataprocessing.transformers.window_transformer import (
    WindowTransformer as DartsWindowTransformer,
)

from ...abstract import AbstractBaseProcessor
from ...time_series import TimeSeries


class Windower(AbstractBaseProcessor):
    """
    Wrapper around Darts WindowTransformer.
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
        """
        A transformer that applies window transformation to a TimeSeries or a Sequence of TimeSeries. It expects a
        dictionary or a list of dictionaries specifying the window transformation(s) to be applied. All series in the
        sequence will be transformed with the same transformations.

        :param transforms: dict or list of dict
            Each dictionary specifies a different window transform. Check the documentation of Darts.
        :param treat_na:
            Specifies how to treat missing values that were added by the window transformations
            at the beginning of the resulting TimeSeries. By default, Darts will leave NaNs in the resulting TimeSeries.
            This parameter can be one of the following: 'dropna', 'backfill', an integer or a float.
        :param forecasting_safe:
            If True, Darts enforces that the resulting TimeSeries is safe to be used in forecasting models as target
            or as feature. The window transformation will not allow future values to be included in the computations
            at their corresponding current timestep. Default is ``True``.
            "ewm" and "expanding" modes are forecasting safe by default.
            "rolling" mode is forecasting safe if ``"center": False`` is guaranteed.
        :param keep_non_transformed:
            ``False`` to return the transformed components only, ``True`` to return all original components along
            the transformed ones. Default is ``False``.
        :param include_current:
            ``True`` to include the current time step in the window, ``False`` to exclude it. Default is ``True``.
        :param name:
            A specific name for the transformer.
        :param n_jobs:
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`.
        :param verbose:
            Whether to print operations progress.
        """
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
        """
        Process the time series.

        :param ts: TimeSeries
        :return: TimeSeries
        """
        return self._processor.transform(ts)
