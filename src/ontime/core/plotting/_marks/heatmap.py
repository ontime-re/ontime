import altair as alt
from altair import Chart

from ...time_series import TimeSeries
from ..plot import Plot


def heatmap(ts: TimeSeries) -> Chart:
    """
    Plot a Heatmap of a TimeSeries

    :param ts: TimeSeries
    :return: Altair Chart
    """
    df = Plot.melt(ts)

    color_condition = alt.condition(
        "month(datum.value) == 1 && date(datum.value) == 1",
        alt.value("black"),
        alt.value(None),
    )

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            alt.X("yearmonthdate(time):O")
            .title("Time")
            .axis(
                format="%Y",
                labelAngle=0,
                labelOverlap=False,
                labelColor=color_condition,
                tickColor=color_condition,
            ),
            alt.Y("variable:N").title(None),
            alt.Color("sum(value)").title("Value"),
        )
    )

    return chart
