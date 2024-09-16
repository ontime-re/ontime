from altair import Chart

from ...time_series import TimeSeries
from ..plot import Plot


def line(ts: TimeSeries, type: str = None, encode_kwargs: dict = None) -> Chart:
    """
    Line plot for TimeSeries

    :param ts: TimeSeries
    :param type: str
    :param encode_kwargs: dict
    :return: Altair Chart
    """

    df = Plot.melt(ts)

    default_kwargs = {
        "x": "time:T",
        "y": "value:Q",
        "color": "variable:N",
    }
    encode_kwargs = encode_kwargs if encode_kwargs is not None else default_kwargs

    match type:
        # Dashed line
        case "dashed":
            chart = (
                Chart(df)
                .mark_line(
                    strokeDash=[5, 2],
                    opacity=0.7,
                )
                .encode(**encode_kwargs)
            )

        # Default line
        case _:
            chart = Chart(df).mark_line().encode(**encode_kwargs)

    return chart
