from typing import Union

import pandas as pd
import lightning as L
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader

from ontime.module.processing.common import train_test_split
from ontime.module.processing.pytorch.utils import create_dataset


class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        series,
        input_length: int,
        target_length: int,
        gap_length: int = 0,
        stride_length: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        test_split: Union[float, int, pd.Timestamp] = 0.0,
        val_split: Union[float, int, pd.Timestamp] = 0.0,
        transform: Pipeline = None,
    ):
        super().__init__()
        # Main dataset
        self.series = series

        # Variables at instantiation
        self.stride_length = stride_length
        self.input_length = input_length
        self.target_length = target_length
        self.gap_length = gap_length

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform
        self.test_split = test_split
        self.val_split = val_split

        # Variables when using the class

        ## Splits
        self.ts_train = None
        self.ts_val = None
        self.ts_test = None

        ## Encoded Splits
        self.ts_train_enc = None
        self.ts_val_enc = None
        self.ts_test_enc = None

        ## Pytorch DataSets
        self.train = None
        self.val = None
        self.test = None
        self.predict = None

    def compute_split(self):
        """
        Splits the series from the datamodule given the parameters defined in the instance
        """
        tmp_ts_train, self.ts_test = train_test_split(
            self.series, test_split=self.test_split
        )
        self.ts_train, self.ts_val = train_test_split(
            tmp_ts_train, test_split=self.val_split
        )

    def compute_transform(self):
        """
        Compute data transformations
        """
        self.transform.fit(self.ts_train)
        self.ts_train_enc = self.transform.transform(self.ts_train)
        self.ts_val_enc = self.transform.transform(self.ts_val)
        self.ts_test_enc = self.transform.transform(self.ts_test)

    def setup(self, stage: str):
        self.compute_split()
        if self.transform is not None:
            self.compute_transform()

        if stage == "fit":
            self.train = create_dataset(
                self.ts_train,
                self.stride_length,
                self.input_length,
                self.target_length,
                self.gap_length,
            )
        if stage == "fit" or stage == "validate":
            self.val = create_dataset(
                self.ts_val,
                self.stride_length,
                self.input_length,
                self.target_length,
                self.gap_length,
            )
        if stage == "test":
            self.test = create_dataset(
                self.ts_test,
                self.stride_length,
                self.input_length,
                self.target_length,
                self.gap_length,
            )
        if stage == "predict":
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        raise NotImplementedError
