from altair import Chart

from ...time_series import TimeSeries
from ..plot import Plot


def dots(ts: TimeSeries, type: str = None, encode_kwargs: dict = None) -> Chart:
    """
    Plot dots of for TimeSeries

    :param ts: TimeSeries,
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
        # Dots (filled circle)
        case "circle":
            chart = Chart(df).mark_circle(filled=False).encode(**encode_kwargs)

        # Square
        case "square":
            chart = Chart(df).mark_square().encode(**encode_kwargs)

        # Default dots (filled circle)
        case _:
            chart = Chart(df).mark_circle().encode(**encode_kwargs)

    return chart
