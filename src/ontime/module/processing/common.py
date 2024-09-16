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


def split_in_windows(ts: TimeSeries, window_length: int, stride_length: int) -> list:
    """
    Split a TimeSeries into parts of a given length, thus forming a window.

    When making the last window, some values of the input TimeSeries might be
    dropped so that all windows have the same length.

    :param ts: TimeSeries to split
    :param window_length: int length of each window
    :param stride_length: int distance in step between the start of the current window and the next one
    :return: list of TimeSeries
    """
    # Check stride_length
    assert stride_length != 0, "stride_length can't be equal to 0, minimum is 1."

    # Get DataFrame
    df = ts.pd_dataframe()

    # Initialize a list to hold the DataFrame splits
    splits_df = []

    # Loop through the DataFrame and split it
    for i in range(len(df.index)):
        start_index = stride_length * i
        end_index = start_index + window_length
        if end_index <= len(df.index):
            splits_df.append(df.iloc[start_index:end_index])

    # Change the data structure from DataFrame to TimeSeries
    return list(map(TimeSeries.from_dataframe, splits_df))


def split_inputs_from_targets(
    ts_list: list,
    input_length: int,
    target_length: int,
    gap_length: int = 0,
) -> tuple:
    """
    Split a list of TimeSeries into input and target TimeSeries

    :param ts_list: list of TimeSeries
    :param input_length: int length of the input TimeSeries
    :param target_length: int length of the target TimeSeries
    :param gap_length: int length of the gap between input end and target start
    :return: tuple of list of TimeSeries
    """
    # Change inner data structure to DataFrame
    dfs = [ts.pd_dataframe() for ts in ts_list]

    # Create initial arrays
    input_series_list = []
    target_series_list = []

    # Iterate over each DataFrame in the list
    for df in dfs:
        # Check
        assert input_length + target_length + gap_length <= len(
            df
        ), "input_length + target_length + gap_length is longer that the total length of the DataFrame."

        # Create inputs
        input_series = df.iloc[:input_length]
        input_series_list.append(input_series)

        # Create targets
        start_target = input_length + gap_length
        end_target = start_target + target_length
        target_series = df.iloc[start_target:end_target]
        target_series_list.append(target_series)

    input_ts_list = list(map(TimeSeries.from_dataframe, input_series_list))
    target_ts_list = list(map(TimeSeries.from_dataframe, target_series_list))

    return input_ts_list, target_ts_list


def timeseries_list_to_numpy(ts_list: list) -> np.array:
    """
    Convert a list of TimeSeries into a numpy array

    :param ts_list: list of TimeSeries
    :return: np.array
    """
    return np.array([ts.pd_dataframe().T.to_numpy() for ts in ts_list])
