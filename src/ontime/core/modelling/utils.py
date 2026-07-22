from __future__ import annotations

import pandas as pd

from ..time_series import TimeSeries


def normalize_prediction(pred: TimeSeries, reference: TimeSeries) -> TimeSeries:
    """
    Normalize a prediction so that its format is consistent with the reference
    (input) time series.

    The returned prediction is guaranteed to have:

    - a time index with the same frequency and name as the reference, starting
      at the time step immediately after the last observed point of the reference;
    - component names matching exactly those of the reference.

    :param pred: the raw prediction produced by a model
    :param reference: the time series the prediction was made from
    :return: TimeSeries with normalized time index and component names
    :raises ValueError: if the number of components differs between the
        prediction and the reference
    """
    if pred.n_components != reference.n_components:
        raise ValueError(
            f"Cannot normalize prediction: it has {pred.n_components} component(s) "
            f"while the reference series has {reference.n_components}."
        )

    time_index = pd.date_range(
        start=reference.end_time() + reference.freq,
        periods=len(pred),
        freq=reference.freq,
        name=reference.time_index.name,
    )

    return TimeSeries.from_times_and_values(
        times=time_index,
        values=pred.values(),
        columns=list(reference.components),
    )


def check_prediction_format(pred: TimeSeries, reference: TimeSeries) -> bool:
    """
    Check that a prediction's format is consistent with the reference (input)
    time series, without considering its actual values.

    :param pred: the prediction to check
    :param reference: the time series the prediction was made from
    :return: True if the format is consistent
    :raises ValueError: if index names, component names or frequencies differ,
        or if the prediction does not start immediately after the reference
    """
    if pred.time_index.name != reference.time_index.name:
        raise ValueError(
            f"Indices have different names: "
            f"{pred.time_index.name!r} != {reference.time_index.name!r}"
        )
    if list(pred.components) != list(reference.components):
        raise ValueError(
            f"Components have different items: "
            f"{list(pred.components)} != {list(reference.components)}"
        )
    if pred.freq != reference.freq:
        raise ValueError(f"Frequencies are different: {pred.freq} != {reference.freq}")
    expected_start = reference.end_time() + reference.freq
    if pred.start_time() != expected_start:
        raise ValueError(
            f"Prediction does not start immediately after the reference: "
            f"starts at {pred.start_time()}, expected {expected_start}"
        )
    return True
