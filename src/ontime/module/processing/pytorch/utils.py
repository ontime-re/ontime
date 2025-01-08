import numpy as np
from ontime.module.processing.common import (
    split_in_windows,
    split_inputs_from_targets,
    timeseries_list_to_numpy,
)
from ontime.core.time_series import TimeSeries
from .time_series_dataset import TimeSeriesDataset


def create_dataset(
    ts: TimeSeries,
    stride_length: int,
    input_length: int,
    target_length: int,
    gap_length: int = 0,
):
    window_length = input_length + target_length + gap_length
    ts_list = split_in_windows(ts, window_length, stride_length)
    input_ts_list, target_ts_list = split_inputs_from_targets(
        ts_list,
        input_length=input_length,
        target_length=target_length,
        gap_length=gap_length,
    )
    features = timeseries_list_to_numpy(input_ts_list)
    labels = timeseries_list_to_numpy(target_ts_list)
    ds = TimeSeriesDataset(features, labels)
    return ds


def dataset_to_numpy(dataset: TimeSeriesDataset):
    data_list = []
    labels_list = []
    for features, labels in dataset:
        data_list.append(features)
        labels_list.append(labels)
    data_array = np.array(data_list)
    labels_array = np.array(labels_list)
    return data_array, labels_array
