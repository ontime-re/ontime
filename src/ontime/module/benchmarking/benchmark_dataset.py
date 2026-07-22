import warnings
from typing import List, Optional, Union, Callable, Tuple
from enum import Enum

from ontime.core.time_series.time_series import TimeSeries
from ontime.module.datasets.dataset import Dataset
from sklearn.base import BaseEstimator


class SeriesDomain(str, Enum):
    WEATHER = "weather"
    FINANCE = "finance"
    TRANSPORT = "transport"
    ENERGY = "energy"
    WEB = "web"

    def __str__(self) -> str:
        # Mirrors enum.StrEnum's __str__ (Python 3.11+), needed since the
        # project supports Python 3.10 where StrEnum is not available.
        return self.value


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
        gap: int = 0,
        stride: Optional[int] = None,
        processing_fn: Optional[Callable[[TimeSeries], TimeSeries]] = None,
        target_columns: Optional[List[str]] = None,
        domain: Optional[Union[SeriesDomain, List[SeriesDomain]]] = None,
        test_proportion: float = 0.2,
        train_proportion: Optional[float] = None,
        validation_proportion: Optional[float] = 0.2,
        few_shot_proportions: List[float] = [1.0],
        train_batch_size: int = 16,
        test_batch_size: int = None,
        scaler_type: Optional[type[BaseEstimator]] = None,
    ):
        """
        Initializes a BenchmarkDataset.

        :param name: name of the dataset
        :param ts: onTime time series or onTime ImportedDataset class that can be loaded
        :param input_length: length of the input, i.e. the context to take into account for making a prediction
        :param target_length: length of the target, i.e. the prediction length
        :param gap: gap in the time series between the end of the input and the begining. Default to 0
        :param stride: stride in the time series between two consecutive window. Default to target_length
        :param target_columns: time series columns to be used as target, i.e. features to be predicted, defaults to None
        :param domain: domain of the time series. If a list, should be the same length as the time series number of components. Defaults to None
        :param test_proportion: proportion of the time series to be used for testing
        :param train_proportion: proportion of the time series to be used for training
        :param validation_proportion: proportion of the training time series to be used for validation, default to None. If None, set to test_proportion
        :param few_shot_proportions: proportions of the training time series to be used for few-shot learning trainings and evaluations,
        defaults to [1.0]
        If None, set to (1 - train_proportion).
        :param train_batch_size: batch size for training
        :param test_batch_size: batch size for testing. Deprecated, will be removed in future releases.
        :processing_fn: processing pipeline to apply to entire ts once loaded, only taken into consideration if a ImportedDataset is given, default to None
        :param scaler_type: sklearn scaler class to use for scaling the time series, default to None
        """
        self._ts = ts
        self.input_length = input_length
        self.gap = gap
        self.stride = stride if stride is not None else target_length
        self.target_length = target_length
        self.name = name
        self.processing_fn = processing_fn or (lambda ts: ts)

        # depreciation about train_proportion
        if train_proportion is not None:
            warnings.warn(
                "The 'train_proportion' argument is deprecated and will be removed in a future version."
                "As you set 'train_proportion' argument, 'test_proportion' is ignored and computed as 1.0 - 'train_proportion'."
                "For futur uses, please consider 'test_proportion' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            test_proportion = 1.0 - train_proportion
        self.test_proportion = test_proportion
        if validation_proportion is None:
            validation_proportion = test_proportion
        self.validation_proportion = validation_proportion
        self.few_shot_proportions = few_shot_proportions
        self.train_batch_size = train_batch_size
        if test_batch_size:
            warnings.warn(
                "The 'test_batch_size' argument is deprecated and will be removed in a future version.\n"
                "You can define a test batch size in model configuration instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.test_batch_size = test_batch_size
        # if target columns is None, we use all columns
        if target_columns is None:
            target_columns = list(self.ts.columns)
        if isinstance(domain, list):
            assert len(domain) == len(self.ts.columns)
        self.domain = domain
        self.target_columns = target_columns
        self.scaler_type = scaler_type

    @property
    def ts(self) -> TimeSeries:
        """
        Getter that returns the dataset time series. If not yet loaded, load and process it.

        :return: the time series
        """
        # TODO: not storing ts in object prevent from memory issues, but is not optimal as
        # dataset will be loaded each time we want to retrieve metadata from it.
        # maybe we should give reponsability to the code using the dataset to destroy the object
        # when no more needed ?
        if not isinstance(self._ts, TimeSeries):
            return self.processing_fn(self._ts.load())
        return self._ts

    def is_multivariate(self):
        """
        Check if the dataset time series is multivariate

        :return: True if the time series is multivariate, False otherwise
        """
        return self.ts.n_components > 1

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
            stacklevel=2,
        )
        return self.ts

    def get_train_test_split(self) -> Tuple[TimeSeries, TimeSeries]:
        """
        Creates the train and test splits according to `train_proportion` parameter

        :return: a tuple of train and test time series
        """
        return self.ts.split_before(1 - self.test_proportion)

    def get_train_val_split(
        self, train_set: Optional[TimeSeries] = None
    ) -> Tuple[TimeSeries, TimeSeries]:
        """
        Creates the train and validation splits according to `validation_proportion` parameter

        :param train_set: the train set to split, if None, the original train set is used
        :return: a tuple of train and validation time series
        """
        if train_set is not None:
            train_set = train_set
        else:
            train_set, _ = self.get_train_test_split()
        return train_set.split_before(1 - self.validation_proportion)

    def get_input_columns(self) -> List[str]:
        """
        Returns the list of columns used as input only

        :return: the list of input columns
        """
        return list(set(self.ts.columns) - set(self.target_columns))

    def get_domain_string_list(self):
        """
        Returns the list of domain as strings.
        A domain is linked to a time series column.

        :return: the list of domain strings
        """
        if self.domain is None:
            return self.domain
        elif isinstance(self.domain, list):
            return [d.value for d in self.domain]
        else:
            return [self.domain] * len(self.ts.columns)
