from typing import List, Optional

from ontime.core.time_series.time_series import TimeSeries


class BenchmarkDataset:
    def __init__(
        self,
        ts: TimeSeries,
        input_length: int,
        target_length: int,
        gap: int,
        stride: int,
        name: str,
        train_proportion: float = 0.8,
        target_columns: Optional[List[str]] = None,
    ):
        self.ts = ts
        self.input_length = input_length
        self.gap = gap
        self.stride = stride
        self.target_length = target_length
        self.name = name
        self.train_proportion = train_proportion
        # if target columns is None, we use all columns
        if target_columns is None:
            target_columns = list(ts.columns)
        self.target_columns = target_columns

    def is_multivariate(self):
        return self.ts.n_components > 1

    def get_data(self):
        return self.ts

    def get_train_test_split(self):
        return self.ts.split_before(self.train_proportion)

    def get_train_val_split(self):
        train_ts, _ = self.get_train_test_split()
        return train_ts.split_before(self.train_proportion)
