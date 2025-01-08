from torch.utils.data import Dataset
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, data_array, labels_array):
        self.data_array = data_array
        self.labels_array = labels_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, index):
        """
        Retrieve the sample and its label at a given index.

        Parameters:
        index (int): The index of the sample to retrieve.

        Returns:
        tuple: containing the sample and its label, both converted to PyTorch tensors.
        """
        # Convert data and labels to PyTorch tensors
        data_tensor = torch.tensor(
            self.data_array[index], dtype=torch.float32
        ).transpose(-1, -2)
        label_tensor = torch.tensor(
            self.labels_array[index], dtype=torch.float32
        ).transpose(-1, -2)

        return data_tensor, label_tensor
