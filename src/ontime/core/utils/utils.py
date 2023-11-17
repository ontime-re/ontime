import pandas as pd


def get_number_of_containing_entries(freq_1, freq_2):
    """
    Calculate the number of entries of a date range object that
    are contained in another date range object.

    :param freq_1: str of a Pandas offset alias
    :param freq_2: str of a Pandas offset alias
    :return: Integer
    """
    # Create date range objects
    start_date = "2000-01-01"
    end_date = "2010-01-01"
    range_1 = pd.date_range(start=start_date, end=end_date, freq=freq_1)
    range_2 = pd.date_range(start=start_date, end=end_date, freq=freq_2)

    # Calculate number of containing entries
    if len(range_1) < len(range_2):
        res = len(range_2) / len(range_1)
    else:
        res = len(range_1) / len(range_2)

    return round(res)
