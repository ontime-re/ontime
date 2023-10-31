from ontime.time_series import TimeSeries
from enum import Enum
import pandas as pd


class Profiler:
    """
    This class should not be instantiated.
    This class is used to make a profile from a time series.
    """

    # Define const for column's names
    VALUE_COL = "value"
    TIME_COL = "time"

    # Define the all aggregations possible
    class Aggregation(Enum):
        MEAN = "mean"
        MEDIAN = "median"
        SUM = "sum"

    # Define the all periods possible
    # The first element is the offset alias for split_by_period from ontime.time_series
    # The second element is the format to make the aggregation (**Also users' format**)
    # The third element is the format to convert data to match with TimeSeries format and the chosen period
    class Period(Enum):
        DAILY = ["D", "%H:%M", "%H:%M"]
        WEEKLY = ["W", "%A %H:%M", "%d %H:%M"]
        MONTHLY = ["M", "%d %H:%M", "%d %H:%M"]
        YEARLY = ["Y", "%m%d %H:%M", "%m%d %H:%M"]

    @staticmethod
    def profile(ts: TimeSeries, period: Period, aggregation: Aggregation):
        """
        Make a profile from a time series
        It resumes the time series by a period and an aggregation.
        For example, if you have a time series with a value for each minute over a year,
        you can make a profile by period (day, week, month or year) with an aggregation (mean, median or sum).
        It returns a time series with the new values that are the aggregation of the values that appear at
        the same period (hour:minute, day of the week hour:minute, day hour:minute, month day hour:minute).
        The index is the **first occurrence** of the period.

        :param ts: TimeSeries
        :param period: Period
        :param aggregation: Aggregation
        :return: TimeSeries
        """
        # Define the column names
        _time_col = Profiler.TIME_COL
        _value_col = Profiler.VALUE_COL

        time_agg = "time_formatted"

        # Split the time series by week
        split_ts = ts.split_by_period(period.value[0])

        # Create a list of DataFrames from the split time series
        data_frames = []
        for time_series in split_ts:
            df = time_series.pd_dataframe()
            col = df.columns[0]
            value = df[col].values
            temp_df = pd.DataFrame(
                {
                    time_agg: df.index.strftime(period.value[1]),
                    _time_col: df.index,
                    _value_col: value,
                }
            )
            data_frames.append(temp_df)

        # Concatenate all the DataFrames in the list into one DataFrame
        data = pd.concat(data_frames)

        # Group the DataFrame by the day of the week and calculate the mean
        grouped = data.groupby([time_agg]).agg(
            {_value_col: aggregation.value, _time_col: "first"}
        )

        # Reset the index, making 'period' the new index column
        grouped.reset_index(inplace=True)

        # Convert the 'period' column to datetime using the custom format
        grouped[_time_col] = pd.to_datetime(grouped[_time_col], format=period.value[2])

        # Return a TimeSeries from the grouped DataFrame
        return TimeSeries.from_dataframe(
            grouped, time_col=_time_col, value_cols=_value_col
        )
