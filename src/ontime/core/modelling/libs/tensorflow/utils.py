import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.data import Dataset

from ontime.core.time_series import TimeSeries


def create_dataset(ts: TimeSeries, input_length: int, target_length: int, step_length: int, stride_length: int) -> Dataset:
    """
    Create a Tensorflow Dataset given a TimeSeries

    :param ts: the TimeSeries to create a Dataset with
    :param input_length: the length of the input for the model (x)
    :param target_length: the length of the output for the model (y)
    :param step_length: the distance between input end and target start in nb of steps
    :param stride_length: the distance between input start from a sample to the next
    :return:
    """

    # Define features and labels
    X = ts[:-input_length].values().flatten()
    Y = ts[input_length+step_length:].values().flatten()

    # Create features dataset
    input_ds = timeseries_dataset_from_array(
        X,
        targets=None,
        sequence_stride=stride_length,
        batch_size=1,
        sequence_length=input_length
    )

    # Create labels dataset
    target_ds = timeseries_dataset_from_array(
        Y,
        targets=None,
        sequence_stride=stride_length,
        batch_size=1,
        sequence_length=target_length
    )

    # Zip the two datasets in one
    dataset = Dataset.zip(input_ds, target_ds)

    return dataset


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