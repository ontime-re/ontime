import altair as alt
from altair import Chart
import pandas as pd

from ...time_series import TimeSeries
from ..plot import Plot


def mark(
    ts: TimeSeries, data: TimeSeries, type: str = None, encode_kwargs: dict = None
) -> Chart:
    """
    Create a plot with marks on it

    :param ts: Base TimeSeries
    :param data: BinaryTimeSeries of mark locations
    :param type: str
    :return: Altair Chart
    """

    def get_marked_ts(ts, data) -> TimeSeries:
        """
        Internal function to merge two TimeSeries
        where one is the original and the other is
        a binary mask.

        This is useful for dots or highlights

        :param ts: TimeSeries
        :param data: BinaryTimeSeries
        :return: TimeSeries
        """
        ts_tmp = ts.pd_series() * data.pd_series()
        ts_tmp = ts_tmp.replace(0, None)
        ts_tmp.name = data.columns[0]
        return TimeSeries.from_dataframe(ts_tmp.to_frame())

    def get_marked_ranges(data) -> pd.DataFrame:
        """
        Internal function to get the ranges of the marks given
        a BinaryTimeSeries acting as a mask.

        This is useful for background marks

        :param data: BinaryTimeSeries
        :return:
        """
        df = Plot.melt(data)
        df["mark_start"] = (df["value"] == 1) & (df["value"].shift(1) != 1)
        df["mark_end"] = (df["value"] == 1) & (df["value"].shift(-1) != 1)
        mark_ranges = pd.DataFrame(
            {
                "start": df.loc[df["mark_start"], "time"],
                "end": df.loc[df["mark_end"], "time"],
            }
        )
        mark_ranges["start"] = mark_ranges["start"].ffill()
        mark_ranges["end"] = mark_ranges["end"].bfill()
        mark_ranges = mark_ranges.drop_duplicates()
        return mark_ranges

    assert ts.is_univariate, "TimeSeries must be univariate"
    assert data.is_univariate, "TimeSeries must be univariate"

    default_kwargs = {
        "x": "time:T",
        "y": "value:Q",
        "color": "variable:N",
    }

    match type:
        case "dot":
            encode_kwargs = (
                encode_kwargs if encode_kwargs is not None else default_kwargs
            )
            ts = get_marked_ts(ts, data)
            df = Plot.melt(ts)
            chart = Chart(df).mark_circle().encode(**encode_kwargs)

        case "highlight":
            encode_kwargs = (
                encode_kwargs if encode_kwargs is not None else default_kwargs
            )
            ts = get_marked_ts(ts, data)
            df = Plot.melt(ts)
            chart = (
                Chart(df).mark_line(strokeWidth=5, opacity=0.5).encode(**encode_kwargs)
            )

        case "background":
            background_kwargs = {
                "x": "start:T",
                "x2": "end:T",
                "color": "variable:N",
            }
            encode_kwargs = (
                encode_kwargs if encode_kwargs is not None else background_kwargs
            )
            df = get_marked_ranges(data)
            df["variable"] = data.columns[0]
            chart = (
                Chart(df)
                .mark_rect(
                    opacity=0.3,
                )
                .encode(**encode_kwargs)
            )

        # Default
        case _:
            raise ValueError("Invalid type")

    return chart
