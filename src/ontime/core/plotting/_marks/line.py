import altair as alt
from altair import Chart

from ...time_series import TimeSeries
from ..plot import Plot


def line(
    ts: TimeSeries,
    type: str = None,
    encode_kwargs: dict = None,
    subplots: bool = False,
    width: int = None,
    height: int = None,
) -> Chart:
    """
    Line plot for TimeSeries

    :param ts: TimeSeries
    :param type: str
    :param encode_kwargs: dict
    :param subplots: whether to display each variable in its own subplot
    :param width: width of the plot
    :param height: height of the plot
    :return: Altair Chart
    """

    df = Plot.melt(ts)

    default_kwargs = {"x": f"{ts.time_index.name}:T", "y": "value:Q"}
    if not subplots:
        default_kwargs["color"] = "variable:N"
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

    if width is not None or height is not None:
        properties = {}
        if width is not None:
            properties["width"] = width
        if height is not None:
            properties["height"] = height
        chart = chart.properties(**properties)

    if subplots:
        return chart.facet(
            row=alt.Row("variable:N", header=alt.Header(title=None))
        ).resolve_scale(y="independent")

    return chart
