import numpy as np
from tensorflow.data import Dataset
from ontime.module.processing.common import (
    split_in_windows,
    split_inputs_from_targets,
    timeseries_list_to_numpy,
)
from ontime.core.time_series import TimeSeries


def create_dataset(
    ts: TimeSeries,
    window_length: int,
    stride_length: int,
    input_length: int,
    target_length: int,
    gap_length: int = 0,
):
    """
    Create a Tensorflow Dataset given a TimeSeries

    :param ts: the TimeSeries to create a Dataset with
    :param window_length: the length of the full window (input, gap and target)
    :param stride_length: the distance between input start from a sample to the next
    :param input_length: the length of the input for the model (x)
    :param gap_length: the length of the gap between input end and target start
    :param target_length: the length of the output for the model (y)
    :return: Dataset
    """

    ts_list = split_in_windows(ts, window_length, stride_length)
    input_ts_list, target_ts_list = split_inputs_from_targets(
        ts_list,
        input_length=input_length,
        target_length=target_length,
        gap_length=gap_length,
    )
    features = timeseries_list_to_numpy(input_ts_list)
    labels = timeseries_list_to_numpy(target_ts_list)
    ds_feature = Dataset.from_tensor_slices(features)
    ds_labels = Dataset.from_tensor_slices(labels)
    ds = Dataset.zip((ds_feature, ds_labels))
    return ds


def dataset_to_numpy(dataset: Dataset):
    """
    Converts a Tensorflow Dataset in a tuple of Numpy arrays

    :param dataset: Tensorflow Dataset
    :return: Tuple with Numpy array of features and Numpy array of labels
    """
    data_list = []
    labels_list = []
    for features, labels in dataset.unbatch():
        data_list.append(features.numpy())
        labels_list.append(labels.numpy())
    # Concatenate all elements to form a single NumPy array for data and labels
    data_array = np.array(data_list)
    labels_array = np.array(labels_list)
    return data_array, labels_array


def arr_to_ts(arr: np.array) -> TimeSeries:
    """
    Converts on array obtained as output of a model to a TimeSeries

    :param arr: Numpy array
    :return: TimeSeries
    """
    arr = np.concatenate(arr, axis=-1)
    return TimeSeries.from_data(arr.T)


def get_input(dataset: Dataset) -> np.array:
    """
    Get the input from a Tensorflow Dataset

    :param dataset: Tensorflow Dataset
    """
    return np.array([x for x, y in dataset])


def get_target(dataset: Dataset) -> np.array:
    """
    Get the target from a Tensorflow Dataset

    :param dataset: Tensorflow Dataset
    """
    return np.array([y for x, y in dataset])
