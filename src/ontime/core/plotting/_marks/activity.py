import calmap
import matplotlib.pyplot as plt

from ...time_series import TimeSeries


def activity(
    ts: TimeSeries, resampling_method="sum", cmap="YlGn", linewidth=2, figsize=(12, 10)
) -> tuple:
    """
    Plot the activity of a TimeSeries in the style of a GitHub activity plot

    :param ts: TimeSeries
    :param resampling_method:str with sum, mean, median, max, min
    :param cmap: Matplotlib colormap
    :param linewidth: int
    :param figsize: tuple
    :return: Matplotlib Figure and Axes
    """

    fig, ax = plt.subplots(figsize=figsize)

    ax = calmap.yearplot(
        ts.pd_series(),
        cmap=cmap,
        how=resampling_method,
        linewidth=linewidth,
        fillcolor="#ADADAD",
        ax=ax,
    )

    plt.gca().grid(False)
    plt.show()
    return fig, ax
