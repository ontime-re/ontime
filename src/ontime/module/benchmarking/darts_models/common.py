import ontime as on

from typing import List

from ontime.module.processing.common import (
    split_in_windows,
    split_inputs_from_targets,
)

def create_dataset(
    ts: on.TimeSeries,
    stride_length: int,
    context_length: int,
    prediction_length: int,
    gap: int = 0,
) -> dict[str, List[on.TimeSeries]]:
    """
    Create a dataset of ontime TimeSeries in an expanding window style from a given time series. The dataset is a dictionary with two keys: "input" and "label".
    """
    # TODO: This method should be improved as we can have memory issues with large time series (e.g. we should process the time series per batch, using a generator).
    dataset = {"input": [], "label": []}
    ts_list = split_in_windows(ts, context_length+prediction_length+gap, stride_length)
    dataset["input"], dataset["label"] = split_inputs_from_targets(
        ts_list,
        input_length=context_length,
        target_length=prediction_length,
        gap_length=gap,
    )
    
    return dataset