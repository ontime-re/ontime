import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from darts.dataprocessing.transformers import Scaler

from ...core.time_series import TimeSeries


def normalize(
    ts: TimeSeries, type="minmax", return_transformer=False
) -> tuple | TimeSeries:
    """
    Normalize a TimeSeries

    :param ts: TimeSeries to normalize
    :param type: str type of normalization to apply
    :param return_transformer: bool whether to return the transformer
    :return: TimeSeries
    """
    match type:
        case "minmax":
            scaler = MinMaxScaler()
        case "zscore":
            scaler = StandardScaler()
    transformer = Scaler(scaler)
    ts_transformed = transformer.fit_transform(ts)
    if return_transformer:
        return ts_transformed, transformer
    else:
        return ts_transformed


def train_test_split(ts: TimeSeries, test_split=None, train_split=None) -> tuple:
    """
    Split a TimeSeries into train and test sets

    :param ts: TimeSeries to split
    :param test_split: float, int or pd.TimeStamp
    :param train_split: float, int or pd.TimeStamp
    :return: tuple of TimeSeries
    """

    if train_split is not None and test_split is not None:
        raise Exception(
            "Only one of those two parameters can be set : train_split, test_split."
        )

    if train_split is None and test_split is None:
        test_split = 0.25

    # split time series in sub time series : train, test
    if test_split is not None:
        train_set, test_set = ts.split_after(1 - test_split)

    if train_split is not None:
        train_set, test_set = ts.split_after(train_split)

    return train_set, test_set


def split_by_length(ts: TimeSeries, length: int, drop_last: bool = True) -> list:
    """
    Split a TimeSeries into parts of a given length

    :param ts: TimeSeries to split
    :param length: int length of each part
    :param drop_last: bool, whether to drop the last part if it is shorter than n
    :return: list of TimeSeries
    """

    # Get DataFrame
    df = ts.pd_dataframe()

    # Calculate the total number of splits needed
    total_splits = -(-len(df) // length)  # Ceiling division to get the number of parts

    # Initialize a list to hold the DataFrame splits
    splits_df = []

    # Loop through the DataFrame and split it
    for split in range(total_splits):
        start_index = split * length
        end_index = start_index + length
        # Append the part to the list, using slicing with .iloc
        splits_df.append(df.iloc[start_index:end_index])

    # If the last dataframe has a different length, then drop it.
    if drop_last:
        last_df = splits_df[-1]
        second_last = splits_df[-2]
        if len(last_df) != len(second_last):
            splits_df = splits_df[:-1]

    # Change the data structure from DataFrame to TimeSeries
    return list(map(TimeSeries.from_dataframe, splits_df))


def split_inputs_from_targets(
    ts_list: list, input_length: int, target_length: int
) -> tuple:
    """
    Split a list of TimeSeries into input and target TimeSeries

    :param ts_list: list of TimeSeries
    :param input_length: int length of the input TimeSeries
    :param target_length: int length of the target TimeSeries
    :return: tuple of list of TimeSeries
    """

    # Change inner data structure to DataFrame
    dfs = [ts.pd_dataframe() for ts in ts_list]

    # Create initial arrays
    input_series_list = []
    target_series_list = []

    # Iterate over each DataFrame in the list
    for df in dfs:
        # Check if the DataFrame is large enough to accommodate input_length and label_len
        if len(df) >= input_length + target_length:
            # Get the first input_length items
            input_series = df.iloc[:input_length]
            input_series_list.append(input_series)
            # Get the last label_len items
            target_series = df.iloc[-target_length:]
            target_series_list.append(target_series)
        else:
            raise Exception(
                "input_length + label_len is longer that the total length of the DataFrame"
            )

    input_ts_list = list(map(TimeSeries.from_dataframe, input_series_list))
    target_ts_list = list(map(TimeSeries.from_dataframe, target_series_list))

    return input_ts_list, target_ts_list


def timeseries_list_to_numpy(ts_list: list) -> np.array:
    """
    Convert a list of TimeSeries into a numpy array

    :param ts_list: list of TimeSeries
    :return: np.array
    """
    return np.array([ts.pd_dataframe().to_numpy() for ts in ts_list])
