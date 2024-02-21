from torch.utils.data import Dataset
from darts import TimeSeries as dts
from ontime.core.time_series.time_series import TimeSeries
import numpy as np
import pandas as pd
from ontime.module import preprocessing as pp
from typing import Tuple

class SlicedDataset(Dataset):
    def __init__(self, data: TimeSeries , period: int, 
                 labels: TimeSeries = None, transform: bool = False, 
                 target_transform: bool = False): 
        self.data, self.labels = self.slice(data, period, labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray] :
        input = self.data[idx]
        if self.labels is not None:
            output = self.labels[idx]
        else:
            output = self.data[idx]
        return input, output

    def slice(self, data: TimeSeries, period: int, labels: TimeSeries = None) -> Tuple[list, list]:
        sliced_data = pp.common.split_by_length(data, period)
        sliced_data = [elt.pd_dataframe().to_numpy() for elt in sliced_data]
        sliced_labels = None
        if labels is not None:
            sliced_labels = pp.common.split_by_length(labels, period)
            sliced_labels = [elt.pd_dataframe().to_numpy() for elt in sliced_labels]
        return sliced_data, sliced_labels