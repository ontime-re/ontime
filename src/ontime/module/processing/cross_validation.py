from typing import List, Optional, Tuple, Union

import numpy as np

from ...core.time_series import TimeSeries

_STRATEGIES = ("expanding", "sliding", "blocked")


def _resolve_size(size: Union[int, float], n: int, name: str) -> int:
    """
    Resolve a size parameter expressed either as an absolute number of points (int)
    or as a fraction of the total length (float in (0, 1)) into an absolute number
    of points.

    :param size: int number of points, or float fraction of `n`
    :param n: int total length of the reference TimeSeries
    :param name: str name of the parameter, used for error messages
    :return: int resolved size
    """
    if isinstance(size, float):
        if not 0 < size < 1:
            raise ValueError(f"When '{name}' is a float, it must be in (0, 1).")
        return max(1, int(round(size * n)))
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"'{name}' must be a positive int.")
        return size
    raise TypeError(f"'{name}' must be an int or a float.")


def cross_validation(
    ts: TimeSeries,
    n_splits: int,
    strategy: str = "expanding",
    initial_train_size: Optional[Union[int, float]] = None,
    horizon: int = 1,
    gap: int = 0,
    stride: Optional[int] = None,
) -> List[Tuple[TimeSeries, TimeSeries]]:
    """
    Split a TimeSeries into a list of (train, test) folds for time series cross-validation.

    Three strategies are supported:

    - ``"expanding"``: the train set is anchored at the start of the series and grows at each
      fold, while the test set slides forward. Setting ``horizon=1`` yields walk-forward
      validation; setting ``horizon`` to more than 1 yields multi-horizon evaluation.
    - ``"sliding"``: the train set has a fixed size and slides forward at each fold together
      with the test set (rolling window).
    - ``"blocked"``: the series is partitioned into ``n_splits + 1`` contiguous, non-overlapping
      blocks of equal size. For each fold, one block is used as train and the next one as test,
      which avoids the train set growing across folds and limits leakage from far away blocks.

    In every strategy, a ``gap`` can be set between the end of the train set and the start of
    the test set to implement purged / embargoed cross-validation, guarding against leakage due
    to autocorrelation.

    :param ts: TimeSeries to split
    :param n_splits: int number of folds to generate
    :param strategy: str one of "expanding", "sliding", "blocked"
    :param initial_train_size: int or float, size of the (first) train set. If a float in (0, 1),
        it is interpreted as a fraction of the length of `ts`. If None, it is computed
        automatically so that `n_splits` folds fit in the series (ignored for "blocked").
    :param horizon: int length of the test set of each fold (ignored for "blocked", where it is
        equal to the block size)
    :param gap: int number of points dropped between the end of the train set and the start of
        the test set of each fold
    :param stride: int number of points between the start of consecutive test sets. Defaults to
        `horizon` (ignored for "blocked")
    :return: list of (train, test) TimeSeries tuples, one per fold
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"'strategy' must be one of {_STRATEGIES}, got '{strategy}'.")
    if n_splits < 1:
        raise ValueError("'n_splits' must be >= 1.")
    if horizon < 1:
        raise ValueError("'horizon' must be >= 1.")
    if gap < 0:
        raise ValueError("'gap' must be >= 0.")

    n = len(ts)

    if strategy == "blocked":
        return _blocked_folds(ts, n, n_splits, gap)

    stride = horizon if stride is None else stride
    if stride < 1:
        raise ValueError("'stride' must be >= 1.")

    if initial_train_size is None:
        initial_train_size = n - gap - horizon - (n_splits - 1) * stride
        if initial_train_size <= 0:
            raise ValueError(
                "Could not infer a valid 'initial_train_size' for the given 'n_splits', "
                "'horizon', 'gap' and 'stride'; the TimeSeries is too short. Provide "
                "'initial_train_size' explicitly or reduce 'n_splits'/'horizon'/'gap'."
            )
    else:
        initial_train_size = _resolve_size(initial_train_size, n, "initial_train_size")

    last_test_end = (
        initial_train_size + gap + horizon + (n_splits - 1) * stride
        if strategy == "expanding"
        else initial_train_size + (n_splits - 1) * stride + gap + horizon
    )
    if last_test_end > n:
        raise ValueError(
            f"The requested cross-validation configuration requires {last_test_end} points "
            f"but the TimeSeries only has {n}. Reduce 'n_splits', 'horizon', 'gap' or "
            f"'initial_train_size'/'stride'."
        )

    if strategy == "expanding":
        return _expanding_folds(ts, n_splits, initial_train_size, horizon, gap, stride)
    return _sliding_folds(ts, n_splits, initial_train_size, horizon, gap, stride)


def _expanding_folds(
    ts: TimeSeries,
    n_splits: int,
    initial_train_size: int,
    horizon: int,
    gap: int,
    stride: int,
) -> List[Tuple[TimeSeries, TimeSeries]]:
    folds = []
    for i in range(n_splits):
        train_end = initial_train_size + i * stride
        test_start = train_end + gap
        test_end = test_start + horizon
        folds.append((ts[0:train_end], ts[test_start:test_end]))
    return folds


def _sliding_folds(
    ts: TimeSeries,
    n_splits: int,
    train_size: int,
    horizon: int,
    gap: int,
    stride: int,
) -> List[Tuple[TimeSeries, TimeSeries]]:
    folds = []
    for i in range(n_splits):
        train_start = i * stride
        train_end = train_start + train_size
        test_start = train_end + gap
        test_end = test_start + horizon
        folds.append((ts[train_start:train_end], ts[test_start:test_end]))
    return folds


def _blocked_folds(
    ts: TimeSeries, n: int, n_splits: int, gap: int
) -> List[Tuple[TimeSeries, TimeSeries]]:
    block_size = n // (n_splits + 1)
    if block_size <= gap:
        raise ValueError(
            f"'gap' ({gap}) is too large for the resulting block size ({block_size}) given "
            f"'n_splits' ({n_splits}) and the length of the TimeSeries ({n}). Reduce 'n_splits' "
            "or 'gap'."
        )
    folds = []
    for i in range(n_splits):
        train_start = i * block_size
        train_end = train_start + block_size
        test_start = train_end + gap
        test_end = test_start + block_size
        folds.append((ts[train_start : train_end - gap], ts[test_start:test_end]))
    return folds


def evaluate_cross_validation(
    model,
    ts: TimeSeries,
    metrics: Union["object", List["object"]],
    n_splits: int,
    strategy: str = "expanding",
    initial_train_size: Optional[Union[int, float]] = None,
    horizon: int = 1,
    gap: int = 0,
    stride: Optional[int] = None,
    refit: bool = True,
) -> dict:
    """
    Evaluate a model with time series cross-validation.

    For each fold generated by :func:`cross_validation`, the model is fit (or refit) on the
    train set and used to predict `horizon` steps, which are then compared to the test set
    using the given metric(s).

    :param model: a model exposing `fit(ts)` and `predict(n, ts)`, such as
        `ontime.core.modelling.Model`
    :param ts: TimeSeries to run the cross-validation on
    :param metrics: a `BenchmarkMetric` (or object exposing `.compute(target, pred)` and
        `.name`), or a list thereof
    :param n_splits: int number of folds, see :func:`cross_validation`
    :param strategy: str one of "expanding", "sliding", "blocked", see :func:`cross_validation`
    :param initial_train_size: see :func:`cross_validation`
    :param horizon: int number of steps to predict and evaluate at each fold
    :param gap: int see :func:`cross_validation`
    :param stride: int see :func:`cross_validation`
    :param refit: bool whether to refit the model on the train set of each fold. If False, the
        model is only fit once, on the train set of the first fold.
    :return: dict with keys "folds" (list of per-fold metric values, in the same order as the
        folds) and "mean"/"std" (aggregated metric values across folds), for each metric name
    """
    if not isinstance(metrics, (list, tuple)):
        metrics = [metrics]

    folds = cross_validation(
        ts,
        n_splits=n_splits,
        strategy=strategy,
        initial_train_size=initial_train_size,
        horizon=horizon,
        gap=gap,
        stride=stride,
    )

    per_fold_results = {metric.name: [] for metric in metrics}
    for i, (train, test) in enumerate(folds):
        if refit or i == 0:
            model.fit(train)
        pred = model.predict(len(test), ts=train)
        for metric in metrics:
            per_fold_results[metric.name].append(
                metric.compute(test, pred, insample=train)
            )

    results = {"folds": per_fold_results}
    results["mean"] = {
        name: np.nanmean(values) for name, values in per_fold_results.items()
    }
    results["std"] = {
        name: np.nanstd(values) for name, values in per_fold_results.items()
    }
    return results
