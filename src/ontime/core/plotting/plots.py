import pandas as pd
import altair as alt

from ..time_series import TimeSeries


def line(ts: TimeSeries) -> alt.Chart:
    """
    Standard line plot for TimeSeries

    :param ts: TimeSeries
    :return: Altair Chart
    """
    alt.data_transformers.enable("vegafusion")

    # Transform data
    df = ts.pd_dataframe()
    df = df.reset_index()
    df = df.melt("time", var_name="variable", value_name="value")

    # Plot
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            color="variable:N",
        )
        .properties(width=500, height=300)
    )

    return chart


def heatmap(ts: TimeSeries) -> alt.Chart:
    """
    Plot a Heatmap of a TimeSeries

    :param ts: TimeSeries
    :return: Altair Chart
    """
    alt.data_transformers.enable("vegafusion")

    # Transform data
    df = ts.pd_dataframe()
    df = df.reset_index()
    df = df.melt("time", var_name="variable", value_name="value")

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
        .properties(width=500, height=100)
    )

    return chart


def prediction(
    train_ts: TimeSeries, pred_ts: TimeSeries = None, test_ts: TimeSeries = None
) -> alt.Chart:
    """
    Plot a prediction

    :param train_ts: TimeSeries
    :param pred_ts: TimeSeries
    :param test_ts: TimeSeries
    :return: Altair Chart
    """
    alt.data_transformers.enable("vegafusion")

    # Train section
    df_train = train_ts.pd_dataframe()
    df_train = df_train.reset_index()
    df_train = df_train.melt("time", var_name="variable", value_name="value")
    df_train["variable"] = "Training set"

    # Prediction section
    df_pred = pred_ts.pd_dataframe()
    df_pred = df_pred.reset_index()
    df_pred = df_pred.melt("time", var_name="variable", value_name="value")
    df_pred["variable"] = "Prediction"

    # Test section
    df_test = test_ts.pd_dataframe()
    df_test = df_test.reset_index()
    df_test = df_test.melt("time", var_name="variable", value_name="value")
    df_test["variable"] = "Truth"

    df = pd.concat([df_train, df_pred, df_test])

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x="time:T",
            y="value:Q",
            color="variable:N",
            opacity=alt.condition(
                alt.datum.variable == "Truth", alt.value(0.5), alt.value(1.0)
            ),
            strokeDash=alt.condition(
                alt.datum.variable == "Truth",
                alt.value(
                    [5, 2]
                ),  # Dash pattern: 5 units of line followed by 2 units of gap
                alt.value([0]),  # Solid line
            ),
        )
        .properties(width=500, height=300)
    )

    return chart


def anomalies(ts: TimeSeries, ts_anomaly: TimeSeries) -> alt.Chart:
    """
    Plot Anomalies

    :param ts: TimeSeries of the signal
    :param ts_anomaly: TimeSeries of the anomalies
    :return: Altair Chart
    """
    alt.data_transformers.enable("vegafusion")

    df = ts.pd_dataframe()
    df = df.reset_index()
    df = df.melt("time", var_name="variable", value_name="value")
    df["variable"] = "signal"

    df_anomaly = ts_anomaly.pd_dataframe()
    df_anomaly = df_anomaly.reset_index()
    df_anomaly = df_anomaly.melt("time", var_name="variable", value_name="value")
    df_anomaly["variable"] = "anomaly"

    df = pd.concat([df, df_anomaly])
    df = df.pivot(index="time", columns="variable", values="value").reset_index()

    chart_signal = (
        alt.Chart(df)
        .mark_line()
        .encode(x="time:T", y="signal:Q")
        .properties(width=500, height=300)
    )

    chart_anomaly = (
        alt.Chart(df)
        .transform_filter(alt.datum.anomaly == 1)
        .mark_circle(color="red", size=60)
        .encode(
            x="time:T",
            y="signal:Q",
        )
    )

    return chart_signal + chart_anomaly
