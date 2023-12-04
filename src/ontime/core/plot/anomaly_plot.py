import pandas as pd

from ..time_series import TimeSeries, BinaryTimeSeries
import altair as alt


class AnomalyPlot:
    _DATA_COLUMN_NAME = "data"

    @staticmethod
    def plot_anomalies(
        data: TimeSeries,
        point_anomalies: BinaryTimeSeries | None = None,
        contextual_anomalies: BinaryTimeSeries | None = None,
        collective_anomalies: BinaryTimeSeries | None = None,
        seasonal_anomalies: BinaryTimeSeries | None = None,
        cyclical_anomalies: BinaryTimeSeries | None = None,
    ):
        """
        Plot the anomalies of the given time series.
        each kind of anomaly is plotted in a different subplot.

        :param data: TimeSeries data
        :param point_anomalies: BinaryTimeSeries of point anomalies. Will be plotted as points on the graph.
        :param contextual_anomalies: BinaryTimeSeries of contextual anomalies. Will be plotted as a line on the graph.
        :param collective_anomalies: BinaryTimeSeries of collective anomalies. Will be plotted as an area on the graph.
        :param seasonal_anomalies: BinaryTimeSeries of seasonal anomalies. Will be plotted as a line on the graph.
        :param cyclical_anomalies: BinaryTimeSeries of cyclical anomalies. Will be plotted as a line on the graph.

        :return: Return an altair chart with the current time series and the anomalies drawn. Data are plot in blue and
        anomalies in red.
        """
        data_df = data.pd_dataframe()

        chart = (
            alt.Chart(data_df.reset_index())
            .mark_line()
            .encode(
                x="time:T",
                y=alt.Y(
                    f"{AnomalyPlot._DATA_COLUMN_NAME}:Q",
                    axis=alt.Axis(title="Values"),
                ),
            )
            .properties(
                title="Chart representing the data and the anomalies over the time",
                width=600,
                height=400,
            )
        )

        if point_anomalies is not None:
            anomalies_df = point_anomalies.pd_dataframe()
            anomalies_df["anomalies_y"] = (
                anomalies_df["anomalies"] * data_df[AnomalyPlot._DATA_COLUMN_NAME]
            )

            anomalies_chart = (
                alt.Chart(anomalies_df.reset_index())
                .mark_circle(color="red", size=100)
                .encode(
                    x="time:T",
                    y="anomalies_y:Q",
                )
                .transform_filter(alt.datum.anomalies == 1)
            )

            chart += anomalies_chart

        if contextual_anomalies is not None:
            chart = AnomalyPlot._make_line_chart(data_df, chart, contextual_anomalies)

        if collective_anomalies is not None:
            chart = AnomalyPlot._make_area_chart(data_df, chart, collective_anomalies)

        if seasonal_anomalies is not None:
            chart = AnomalyPlot._make_line_chart(data_df, chart, seasonal_anomalies)

        if cyclical_anomalies is not None:
            chart = AnomalyPlot._make_line_chart(data_df, chart, cyclical_anomalies)

        return chart

    @staticmethod
    def _make_line_chart(
        data: pd.DataFrame, actual_chart: alt.Chart, anomalies: BinaryTimeSeries
    ) -> alt.Chart:
        """
        Make a line chart with the given anomalies.

        :param data: TimeSeries within the data that are plotted in the chart.
        :param actual_chart: Chart that will be updated with the anomalies.
        :param anomalies: BinaryTimeSeries of anomalies that will be plotted in red on the chart.

        :return: Return an altair chart with the current time series and the anomalies drawn.
        """
        array_anomalies_df = AnomalyPlot.split_continuous_series(anomalies)
        chart_total = actual_chart
        for anomalies_df in array_anomalies_df:
            anomalies_df["anomalies_y"] = (
                anomalies_df["anomalies"] * data[AnomalyPlot._DATA_COLUMN_NAME]
            )
            chart = (
                alt.Chart(anomalies_df.reset_index())
                .mark_line(color="red", strokeWidth=2.5)
                .encode(
                    x="time:T",
                    y="anomalies_y:Q",
                )
            )
            chart_total += chart

        return chart_total

    @staticmethod
    def _make_area_chart(
        data_df: pd.DataFrame, actual_chart: alt.Chart, anomalies: BinaryTimeSeries
    ) -> alt.Chart:
        """
        Adding background to the chart with the given anomalies.

        :param data_df: TimeSeries within the data that are plotted in the chart.
        :param actual_chart: Chart that will be updated with the anomalies.
        :param anomalies: BinaryTimeSeries of anomalies that will be used to color the background in red on the chart.

        :return: Return an altair chart with the current time series and the anomalies drawn.
        """

        y_max = data_df[AnomalyPlot._DATA_COLUMN_NAME].max()
        y_min = data_df[AnomalyPlot._DATA_COLUMN_NAME].min()

        delta = (y_max - y_min) * 0.1

        array_anomalies_df = AnomalyPlot.split_continuous_series(anomalies)
        chart_total = None
        for anomalies_df in array_anomalies_df:
            anomalies_df["max"] = y_max + delta
            anomalies_df["min"] = y_min - delta

            chart = (
                alt.Chart(anomalies_df.reset_index())
                .mark_area(color="red", opacity=0.3)
                .encode(
                    x="time:T",
                    y="min:Q",
                    y2="max:Q",
                )
            )

            if chart_total is None:
                chart_total = chart
            else:
                chart_total += chart

        if chart_total is not None:
            # Put our chart as background
            return chart_total + actual_chart
        return actual_chart

    @staticmethod
    def split_continuous_series(anomalies: BinaryTimeSeries) -> list[pd.DataFrame]:
        """
        Split a continuous series of anomalies into multiple series of anomalies. The response covert all the anomalies
        in the BinaryTimeSeries but change its structure to have a list with one df by continuous series of 1. Zeros are
        not represented anymore.

        :param anomalies: BinaryTimeSeries of anomalies that will be split.

        :return: Return a list of DataFrame of anomalies.
        """
        anomalies_df = anomalies.pd_dataframe()

        # Initialize variables
        result_dfs = []
        current_value = 0
        current_df = None

        # Iterate through each row of the base_df
        for idx, row in anomalies_df.iterrows():
            value = row["anomalies"]

            # If value changes
            if value != current_value:
                if current_df is not None:
                    # Save the current DataFrame into the array
                    current_df = current_df.rename_axis("time")
                    result_dfs.append(current_df)
                    current_df = None

                else:
                    # Create a new DataFrame with the same structure as base_df
                    current_df = pd.DataFrame(columns=anomalies_df.columns)
                    # Save the current row in the new DataFrame
                    current_df = pd.concat(
                        [current_df, pd.DataFrame([row], index=[idx])]
                    )

                current_value = value

            else:
                # If value does not change
                if current_df is not None:
                    # Save the row in the current DataFrame
                    current_df = pd.concat(
                        [current_df, pd.DataFrame([row], index=[idx])]
                    )

        if current_value == 1:
            current_df = current_df.rename_axis("time")
            result_dfs.append(current_df)

        return result_dfs
