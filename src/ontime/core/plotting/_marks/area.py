import altair as alt
from altair import Chart

from ...time_series import TimeSeries
from ..plot import Plot


def area(ts: TimeSeries, title: str = "Area", encode_kwargs: dict = None) -> Chart:
    # Define type given the number of components
    if ts.is_univariate:
        type = "single"
    elif len(ts.components) == 2:
        type = "double"
    else:
        raise ValueError("TimeSeries must be univariate or have exactly two components")

    df = Plot.melt(ts)

    match type:
        case "single":
            assert ts.is_univariate, "TimeSeries must be univariate"

            # Manually add the title
            df["area"] = title

            # Define kwargs
            default_kwargs = {
                "x": "time:T",
                "y": "value:Q",
                "color": alt.Color("area:N", legend=alt.Legend(title="variable")),
            }
            encode_kwargs = (
                encode_kwargs if encode_kwargs is not None else default_kwargs
            )

            # Make the chart
            chart = (
                Chart(df).mark_area(opacity=0.6, color="gray").encode(**encode_kwargs)
            )

        case "double":
            assert (
                len(ts.components) == 2
            ), "TimeSeries must have exactly two components"

            # Pivot the data and manually add the title
            df = df.pivot(
                index="time", columns="variable", values="value"
            ).reset_index()
            df["area"] = title

            # Define kwargs
            default_kwargs = {
                "x": "time:T",
                "y": f"{ts.columns[0]}:Q",
                "y2": f"{ts.columns[1]}:Q",
                "color": alt.Color("area:N", legend=alt.Legend(title="variable")),
            }
            encode_kwargs = (
                encode_kwargs if encode_kwargs is not None else default_kwargs
            )

            # Make the chart
            chart = (
                Chart(df).mark_area(opacity=0.6, color="gray").encode(**encode_kwargs)
            )

    return chart
