from ..time_series import TimeSeries, BinaryTimeSeries
import matplotlib.pyplot as plt

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
        fig, axs = plt.subplots( figsize=(20, 10))

        axs.plot(data)

        return fig, axs
