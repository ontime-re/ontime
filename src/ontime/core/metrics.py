import numpy as np
from darts import TimeSeries
from darts.metrics.metrics import (
    multi_ts_support,
    METRIC_OUTPUT_TYPE,
    _get_values_or_raise,
    logger,
    raise_log,
    multivariate_support,
    TIME_AX,
    _get_wrapped_metric,
    mae,
    COMP_AX,
    ae,
)
from typing import Union, Optional, Sequence, Callable
import pandas as pd


@multi_ts_support
@multivariate_support
def nmae(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> METRIC_OUTPUT_TYPE:
    """
    Compute the Normalized Mean Absolute Error (NMAE) as the mean of per-component NMAE values.

    This version computes the NMAE for each individual component (feature/series),
    then returns the mean of those values. This ensures scale invariance across components.

    Formula:
        NMAE = (1 / K) * sum_{k=1}^K [ sum_{t=1}^T |x_t^k - x̂_t^k| / sum_{t=1}^T |x_t^k| ]

    :param actual_series: The (sequence of) actual series.
    :param pred_series: The (sequence of) predicted series.
    :param intersect: For time series that are overlapping in time without having the same time index, setting `True` will consider the values only over their common time interval (intersection in time). Defaults to True.
    :param component_reduction: a function to aggregate the metrics over the component/column axis. It must reduce a `np.ndarray` of shape `(t, c)` to a `np.ndarray` of shape `(t,)`. The function takes as input a ``np.ndarray`` and a parameter named `axis`, and returns the reduced array. The `axis` receives value `1` corresponding to the component axis. If `None`, will return a metric per component. Defaults to np.nanmean
    :param series_reduction: Optionally, a function to aggregate the metrics over the series axis. It must reduce a `np.ndarray` of shape `(s, t, c)` to a `np.ndarray` of shape `(t, c)` The function takes as input a ``np.ndarray`` and a parameter named `axis`, and returns the reduced array. The `axis` receives value `0` corresponding to the series axis. If `None`, will return a metric per series. Defaults to None
    :param n_jobs: The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available processors.
    :param verbose: Optionally, whether to print operations progress. Defaults to False
    :return:
    float
        A single metric score for:

        - single univariate series.
        - single multivariate series with `component_reduction`.
        - a sequence (list) of uni/multivariate series with `series_reduction`, `component_reduction` and
          `time_reduction`.
    np.ndarray
        A numpy array of metric scores. The array has shape (n time steps, n components) without time
        and component reductions. For:

        - single multivariate series and at least `component_reduction=None`.
        - single uni/multivariate series and at least `time_reduction=None`.
        - a sequence of uni/multivariate series including `series_reduction` and at least one of
          `component_reduction=None` or `time_reduction=None`.
    List[float]
        Same as for type `float` but for a sequence of series.
    List[np.ndarray]
        Same as for type `np.ndarray` but for a sequence of series.
    """

    y_true, _ = _get_values_or_raise(
        actual_series, pred_series, intersect, remove_nan_union=True
    )

    y_true_sum = np.nansum(y_true, axis=TIME_AX)

    if not (y_true_sum > 0).all():
        raise_log(
            ValueError(
                "The series of actual value cannot sum to zero when computing OPE."
            ),
            logger=logger,
        )

    return (
        np.nansum(
            _get_wrapped_metric(ae)(
                actual_series, pred_series, intersect, time_reduction=np.nansum
            ),
            axis=TIME_AX,
        )
        / y_true_sum
    )
