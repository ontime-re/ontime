from typing import List, Optional

from ontime.core.time_series.time_series import TimeSeries

class BenchmarkDataset:
    def __init__(self, ts: TimeSeries, input_length: int, gap: int, stride: int, horizon: int, name: str, target_columns: Optional[List[str]] = None):
        self.ts = ts
        self.input_length = input_length
        self.gap = gap
        self.stride = stride
        self.horizon = horizon
        self.name = name
        # if target columns is None, we use all columns
        if target_columns is None:
            target_columns = list(ts.columns)
        self.target_columns = target_columns
        
    def is_multivariate(self):
        return self.ts.n_components > 1
    
    def get_data(self):
        return self.ts

    def get_train_test_split(self, train_proportion=0.7):
        return self.ts.split_before(train_proportion)