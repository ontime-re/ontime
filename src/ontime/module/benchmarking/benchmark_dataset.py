import warnings
from typing import List, Optional, Union, Callable, Tuple

from ontime.core.time_series.time_series import TimeSeries
from ontime.module.datasets.dataset import Dataset


class BenchmarkDataset:
    """
    BenchmarkDataset class that holds a time series and processing parameter to use it when training and evaluating a model (benchmarking).
    """
    def __init__(
        self,
        ts: Union[TimeSeries, Dataset.ImportedDataset],
        name: str,
        input_length: int,
        target_length: int,
        gap: int,
        stride: int,
        processing_fn: Optional[Callable[[TimeSeries], TimeSeries]] = None,
        target_columns: Optional[List[str]] = None,
        train_proportion: float = 0.8,
        validation_proportion: Optional[float] = None,
        train_batch_size: int = 16,
        test_batch_size: int = 16,
    ):
        """
        Initializes a BenchmarkDataset.

        :param name: name of the dataset
        :param ts: onTime time series or onTime ImportedDataset class that can be loaded
        :param input_length: length of the input, i.e. the context to take into account for making a prediction
        :param target_length: length of the target, i.e. the prediction length
        :param gap: gap in the time series between the end of the input and the begining
        :param stride: stride in the time series between two consecutive window
        :param target_columns: time series columns to be used as target, i.e. features to be predicted, defaults to None
        :param train_proportion: proportion of the time series to be used for training
        :param validation_proportion: proportion of the training time series to be used for validation, defaults to None. 
        If None, set to (1 - train_proportion).
        :param train_batch_size: batch size for training
        :param test_batch_size: batch size for testing
        :processing_fn: processing pipeline to apply to entire ts once loaded, only taken into consideration if a ImportedDataset is given, default to None
        """
        self.ts = ts
        self.input_length = input_length
        self.gap = gap
        self.stride = stride
        self.target_length = target_length
        self.name = name
        self.train_proportion = train_proportion
        self.validation_proportion = validation_proportion
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        # if target columns is None, we use all columns
        if target_columns is None:
            target_columns = list(ts.columns)
        self.target_columns = target_columns
        self.processing_fn = processing_fn or (lambda ts: ts)
        
    def get_ts(self) -> TimeSeries:
        """
        Returns the dataset time series. If not yet loaded, load and process it.
        
        :return: the time series
        """
        # TODO : maybe we don't want necessarily to store the instantiated ts in this class ?
        if not isinstance(self.ts, TimeSeries):
            self.ts = self.processing_fn(self.ts.load())
        return self.ts

    def is_multivariate(self):
        """
        Check if the dataset time series is multivariate

        :return: True if the time series is multivariate, False otherwise
        """
        return self.get_ts().n_components > 1

    def get_data(self):
        """
        Deprecated: Use `get_ts`instead
        Returns the dataset time series 

        :return: the time series
        """
        warnings.warn(
            "`BenchmarkDataset.get_data()` is deprecated and will be removed in future releases."
            "Use BenchmarkDataset.get_ts() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_ts()

    def get_train_test_split(self) -> Tuple[TimeSeries, TimeSeries]:
        """
        Creates the train and test splits according to `train_proportion` parameter

        :return: a tuple of train and test time series
        """
        return self.get_ts().split_before(self.train_proportion)

    def get_train_val_split(self):
        """
        Creates the train and validation splits according to `validation_proportion` parameter

        :return: a tuple of train and validation time series
        """
        train_ts, _ = self.get_train_test_split()
        return train_ts.split_before(self.train_proportion)
    
    def get_input_columns(self):
        """
        Returns the list of columns used as input only
        
        :return: the list of input columns
        """
        return list(set(self.ts.columns) - set(self.target_columns))
        
        
