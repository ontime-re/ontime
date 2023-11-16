import pandas as pd

from ..time_series import TimeSeries, BinaryTimeSeries
import altair as alt


class AnomalyTimeSeries:
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
        :param seasonal_anomalies: BinaryTimeSeries of seasonal anomalies. TODO
        :param cyclical_anomalies: BinaryTimeSeries of cyclical anomalies. TODO

        :return: TODO
        """
        data_df = data.pd_dataframe()

        chart = alt.Chart(data_df.reset_index()).mark_line().encode(
            x='time:T',
            y='random_walk:Q',
        ).properties(
            title='Line Chart',
            width=600,
            height=400
        )

        if point_anomalies is not None:
            anomalies_df = point_anomalies.pd_dataframe()
            anomalies_df['anomalies_y'] = anomalies_df['anomalies'] * data_df['random_walk']

            anomalies_chart = alt.Chart(anomalies_df.reset_index()).mark_circle(color='red').encode(
                x='time:T',
                y='anomalies_y:Q',
            ).transform_filter(
                alt.datum.anomalies == 1
            )

            chart += anomalies_chart

        if contextual_anomalies is not None:
            chart += AnomalyTimeSeries._make_line_chart(data_df, contextual_anomalies)

        if collective_anomalies is not None:
            chart += AnomalyTimeSeries._make_line_chart(data_df, collective_anomalies)

        if seasonal_anomalies is not None:
            chart += AnomalyTimeSeries._make_line_chart(data_df, seasonal_anomalies)

        if cyclical_anomalies is not None:
            chart += AnomalyTimeSeries._make_line_chart(data_df, cyclical_anomalies)

        return chart

    @staticmethod
    def _make_line_chart(data: pd.DataFrame, anomalies_ts: BinaryTimeSeries):
        anomalies = anomalies_ts.pd_dataframe()
        anomalies['anomalies_y'] = anomalies['anomalies'] * data['random_walk']

        return alt.Chart(anomalies.reset_index()).mark_line(color='red').encode(
            x='time:T',
            y='anomalies_y:Q',
        )
