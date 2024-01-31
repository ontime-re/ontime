from torch.utils.data import Dataset
from darts import TimeSeries as dts

class VAEDataset(Dataset):
    def __init__(self, data , period, labels = None, transform=False, target_transform=False): #data = a pandas dataframe
        self.data, self.labels = self.slice(data, period, labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx]
        if self.labels is not None:
            output = self.labels[idx]
        else:
            output = self.data[idx]
        return input, output

    def slice(self, data, period, labels = None):
        if (isinstance(data,dts)):
            index = data.time_index.tolist()
        else:
            index = data.index.tolist()
        sliced_data = []
        sliced_labels = []
        remaining_data = data
        remaining_labels = labels
        for i in range(period, len(index), period):
            if isinstance(data, dts):
                slice_stop = index[i]
                slice, remaining_data = remaining_data.split_before(slice_stop)
                sliced_data.append(slice.pd_dataframe().to_numpy())
            else: #pandas dataframe
                slice = remaining_data.iloc[:period,:]
                remaining_data = remaining_data.iloc[:period,:]
                sliced_data.append(slice.to_numpy())
            if labels is not None:
                slice_label, remaining_labels = remaining_labels.split_before(slice_stop)
                if 1 in slice_label.pd_dataframe()['Class'].unique():
                    sliced_labels.append(1)
                else:
                    sliced_labels.append(0)
        if labels is None: sliced_labels = None
        return sliced_data, sliced_labels