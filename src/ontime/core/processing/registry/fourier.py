from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..abstract_processor import AbstractProcessor
from ...time_series import TimeSeries


class Fourier(AbstractProcessor):
    """Fourier class performs a sliding-window FFT (Short-Time Fourier
    Transform) on a TimeSeries.

    Each window of ``window_size`` samples is transformed with a real FFT,
    the resulting amplitude spectrum is aggregated into ``n_bins`` frequency
    ranges, and the window slides by ``step_size`` samples. The result is a
    multivariate TimeSeries where each column is a frequency range and each
    row is a window, timestamped at the window's last sample.
    """

    def __init__(
        self,
        window_size: int,
        step_size: int = 1,
        n_bins: int = 10,
        frequency_cap: Optional[Tuple[float, float]] = None,
    ):
        """Constructor of a Fourier processor

        Frequencies are expressed in cycles per sample, ranging from 0 to
        0.5 (the Nyquist frequency).

        :param window_size: int, number of samples per FFT window
        :param step_size: int, number of samples the window slides by
        :param n_bins: int, number of frequency ranges in the output
        :param frequency_cap: optional (min, max) tuple, in cycles per
            sample, to restrict the frequency range before binning
        """
        assert (
            isinstance(window_size, int) and window_size > 1
        ), f"window_size must be an integer greater than 1, not {window_size}"
        assert (
            isinstance(step_size, int) and step_size > 0
        ), f"step_size must be a positive integer, not {step_size}"
        assert (
            isinstance(n_bins, int) and n_bins > 0
        ), f"n_bins must be a positive integer, not {n_bins}"
        if frequency_cap is not None:
            assert (
                len(frequency_cap) == 2 and 0 <= frequency_cap[0] < frequency_cap[1]
            ), f"frequency_cap must be a (min, max) tuple with 0 <= min < max, not {frequency_cap}"

        self.window_size = window_size
        self.step_size = step_size
        self.n_bins = n_bins
        self.frequency_cap = frequency_cap

    def process(self, ts: TimeSeries) -> TimeSeries:
        """Compute the sliding-window FFT of a TimeSeries

        :param ts: TimeSeries, univariate
        :return: TimeSeries, multivariate, one column per frequency range
        """
        assert ts.width == 1, f"ts must be univariate, but has {ts.width} components"
        values = ts.values().flatten()
        assert len(values) >= self.window_size, (
            f"ts must have at least window_size ({self.window_size}) values, "
            f"but has {len(values)}"
        )

        # Frequencies in cycles per sample for a real FFT of the window
        frequencies = np.fft.rfftfreq(self.window_size, d=1.0)

        f_min, f_max = 0.0, frequencies[-1]
        if self.frequency_cap is not None:
            f_min = max(f_min, self.frequency_cap[0])
            f_max = min(f_max, self.frequency_cap[1])
        mask = (frequencies >= f_min) & (frequencies <= f_max)
        assert mask.any(), (
            f"frequency_cap {self.frequency_cap} excludes all FFT frequencies "
            f"(available range: 0 to {frequencies[-1]})"
        )

        bin_edges = np.linspace(f_min, f_max, self.n_bins + 1)

        starts = range(0, len(values) - self.window_size + 1, self.step_size)
        rows = []
        times = []
        for start in starts:
            window = values[start : start + self.window_size]
            amplitudes = np.abs(np.fft.rfft(window)) / self.window_size
            row = []
            for i in range(self.n_bins):
                in_bin = (
                    mask
                    & (frequencies >= bin_edges[i])
                    & (
                        (frequencies < bin_edges[i + 1])
                        if i < self.n_bins - 1
                        else (frequencies <= bin_edges[i + 1])
                    )
                )
                row.append(amplitudes[in_bin].mean() if in_bin.any() else 0.0)
            rows.append(row)
            times.append(ts.time_index[start + self.window_size - 1])

        columns = [
            f"freq_{bin_edges[i]:.4f}_{bin_edges[i + 1]:.4f}"
            for i in range(self.n_bins)
        ]
        df = pd.DataFrame(rows, index=times, columns=columns)

        return TimeSeries.from_dataframe(df)
