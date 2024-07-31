from ontime.core.time_series.time_series import TimeSeries

from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum
import pandas as pd
import time
import traceback

class BenchmarkMode(Enum):
        ZERO_SHOT = 1 # no training, only inference
        FULL_SHOT = 3 # full training
    
class BenchmarkMetric:
    def __init__(self, name: str, metric_function, component_reduction = None):
        self.metric = metric_function
        self.name = name
        self.component_reduction = component_reduction

    def compute(self, target: TimeSeries, pred: TimeSeries):
        """
        Compute the metric on the target and predicted time series.
        """
        return self.metric(target, pred, component_reduction=self.component_reduction)

class AbstractBenchmarkModel(ABC):
    @abstractmethod
    def fit(self, train_ts: TimeSeries, val_ts: TimeSeries, *args, **kwargs) -> None:
        """
        Fit a model on training data.
        """
        pass

    @abstractmethod
    def predict(self, ts: TimeSeries, horizon: int, *args, **kwargs) -> TimeSeries:
        """
        Predict the next `horizon` steps of the time series.
        """
        pass

    @abstractmethod
    def evaluate(self, ts: TimeSeries, horizon: int, metrics: List[BenchmarkMetric], *args, **kwargs) -> dict:
        """
        Evaluate the model on test data, using the given metrics.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """
        Load a model checkpoint from the given path.
        """
        pass
    
    @abstractmethod
    def get_benchmark_mode(self) -> BenchmarkMode:
        """
        Return the benchmark mode of the model.
        """
        pass

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
    
    def get_data(self):
        return self.ts

    def get_train_test_split(self, train_proportion=0.7):
        return self.ts.split_before(train_proportion)



class Benchmark:                
    def __init__(self,
                 datasets: List[BenchmarkDataset] = None,
                 models: List[AbstractBenchmarkModel] = None,
                 metrics: List[BenchmarkMetric] = None,
                 train_proportion=0.7):

        self.train_proportion = train_proportion
        # contains BenchmarkDatasets
        self.datasets = []
        # contains BenchmarkModelHolders
        self.models = []
        # contains BenchmarkMetrics
        self.metrics = []

        # for holding results
        self.metrics_results = []
        self.results = None

        # initialize datasets, models and metrics
        if metrics is not None:
            self.metrics = metrics
        if datasets is not None:
            i = 1
            for d in datasets:
                self.add_dataset(d, str(i))
                i += 1
        if models is not None:
            for m in models:
                self.add_model(m)

    def add_model(self, model):
        if not isinstance(model, AbstractBenchmarkModel):
            raise TypeError(f"models must implement {AbstractBenchmarkModel}, found{type(model)}.")
        self.models.append(model)

    def get_models(self):
        return [m.model_class for m in self.models]

    def add_dataset(self, dataset: BenchmarkDataset):
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError(
                f"datasets must be of type {BenchmarkDataset}, found type {type(dataset)}")
        self.datasets.append(dataset)

    def get_datasets(self):
        return [d for d in self.datasets]

    def add_metric(self, metric: BenchmarkMetric):
        self.metrics.append(metric)

    def get_metrics(self):
        return self.metrics

    def run(self, verbose: bool = False, debug: bool = False):
        # TODO: throw error if models or datasets is empty
        self.results = {}  # self.results[model name][dataset name] = {metric name: value, metric name: value,...}
        if verbose:
            print("Starting evaluation...")
        i = 0
        for source_model in self.models:
            self.results[source_model.name] = {}
            if verbose:
                print(f"Evaluation for model {source_model.name}")

            # internal metrics (computed during run time)
            mv0 = len([x for x in self.datasets if x.multivariate])
            nb_mv_run_succeeded = mv0
            nb_uv_run_succeeded = len(self.datasets) - mv0

            # evaluation time!
            for dataset in self.datasets:
                    print(f"on dataset {dataset.name}")

                    # initialize variables
                    nb_features = len(dataset.ts.columns)
                    train_set, test_set = dataset.get_train_test_split(self.train_proportion)
                    train_size = len(train_set.time_index)
                    test_size = len(test_set.time_index)
                    train_time = 0
                    inference_time = 0
                    run_success = True
                    try:
                        # train
                        if source_model.get_benchmark_mode() != BenchmarkMode.ZERO_SHOT:
                            if verbose:
                                print(f"training... ", end="")
                            start_time = time.time()
                            source_model.fit(train_set)
                            train_time = time.time() - start_time

                        
                            if verbose:
                                print(f"done, took {train_time}", end="")
                        # test
                        if verbose:
                            print(f"testing... ", end="")
                        start_time = time.time()
                        metrics = source_model.evaluate(test_set, dataset.horizon, self.metrics)
                        inference_time = time.time() - start_time
                        if verbose:
                            print(f"done, took {inference_time}")

                    except:  # can't train or test on this dataset
                        run_success = False
                        if nb_features == 1:
                            nb_uv_run_succeeded -= 1
                        else:
                            nb_mv_run_succeeded -= 1
                        if verbose:
                            print(f"Couldn't complete evaluation.")
                            if debug:
                                traceback.print_exc()

                    if run_success:
                        # compute metrics
                        self.results[source_model.name][dataset.name] = {
                            'nb features': nb_features,
                            'target column': dataset.target_columns,
                            'training set size': train_size,
                            'training time': train_time,
                            'test set size': test_size,
                            'testing time': inference_time,
                        }
                        # compute user-submitted metrics
                        if verbose:
                            print(f"Computed metrics: {metrics}")
                            
                        self.results[source_model.name][dataset.name]['metrics'] = metrics
                  
            # back to model-level stats
            supports_uni = None
            if len(self.datasets) - mv0 > 0:  # if we have univariate ds in our batch
                supports_uni = (nb_uv_run_succeeded > 0)
            supports_mult = None
            if mv0 > 0:
                supports_mult = (nb_mv_run_succeeded > 0)
            self.results[source_model.name]['supports univariate'] = Benchmark._bool_to_symbol(supports_uni)
            self.results[source_model.name]['supports multivariate'] = Benchmark._bool_to_symbol(supports_mult)
            i += 1

    @staticmethod
    def _bool_to_symbol(b: bool) -> str:
        if b is None:
            return 'unknown'
        if b:
            return '✓'
        return 'X'

    def get_report(self):
        if self.results is None:
            return "please invoke run_benchmark() to generate report data"
        txt = []

        for model in self.models:
            s = f"Model {model.name}:\n"
            s += f"Supported univariate datasets: {self.results[model.name]['supports univariate']}\n"
            s += f"Supported multivariate datasets: {self.results[model.name]['supports multivariate']}\n"
            for dataset in self.datasets:
                    s += f"Dataset {dataset.name}:\n"
                    if dataset.name not in self.results[model.name].keys():
                        s += f"dataset.name couldn't complete training\n"
                    else:
                        for key in self.results[model.name][dataset.name].keys():
                            if key == 'prediction':
                                continue
                            s += f"{key}: {self.results[model.name][self.dataset.name][key]}\n"
            txt.append(s)
        report = "\n\n".join(txt)
        return report

    def get_report_dataframes(self) -> dict:
        # DISCLAIMER: this is still bugged
        # 1 dataframe per model
        # index = datasets
        # columns = metric and details
        col0 = self.datasets[0].training_set.columns.tolist()[0]
        me_columns = self.results[self.models[0].name][self._c(self.datasets[0].name, col0)].keys()
        reports = {}
        for model in self.models:
            report_model = pd.DataFrame(columns=me_columns)
            for dataset in self.datasets:
                for target in dataset.target_columns:
                    if self._c(dataset.name, target) in self.results[model.name].keys():
                        report_dict = self.results[model.name][self._c(dataset.name, target)]
                        del report_dict["prediction"]
                        report_model.loc[dataset.name] = report_dict

            reports[model.name] = report_model
        return reports

    def get_results(self):
        return self.results