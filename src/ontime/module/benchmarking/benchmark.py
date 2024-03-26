import numpy as np
from ontime.core.time_series.time_series import TimeSeries
from ontime.core.model.abstract_model import AbstractModel
import pandas as pd
import time
from typing import List, Union, Tuple
import traceback


class Benchmark:
    # a wrapper for datasets used in Benchmark
    class BenchmarkDataset:
        def __init__(self, training_set: TimeSeries, multivariate: bool = None, name: str = None,
                     test_set: TimeSeries = None, *args):
            self.training_set = training_set
            self.test_set = test_set
            if multivariate is None:
                multivariate = len(training_set.columns) > 1
            self.multivariate = multivariate
            self.name = name

        def get_train_test_split(self, train_proportion=0.7):
            if self.test_set is None:
                return self.training_set.split_before(train_proportion)
            return self.training_set, self.test_set

    # a wrapper containing models used in Benchmark.
    # can be used as is or overridden to adapt for particular models
    class BenchmarkModelHolder:
        def __init__(self, model_class, name: str,
                     arguments_dict: dict = None):
            self.model = model_class
            self.model_instance = None
            self.args = arguments_dict
            self.name = name

        def instantiate(self, train_set: TimeSeries, test_set: TimeSeries, ):
            self.model_instance = self.model(**self.args)

        def fit(self, training_set: TimeSeries, test_set: TimeSeries):
            self.model_instance.fit(training_set)

        def predict(self, horizon, test_set: TimeSeries):
            return self.model_instance.predict(horizon)

        def name(self):
            return self.name

    # wrapper for user-submitted metrics to be used in Benchmark
    class BenchmarkMetric:
        def __init__(self, metric_function, name: str):
            self.metric = metric_function
            self.name = name

        def compute(self, data: TimeSeries, pred: TimeSeries):
            return self.metric(data, pred)

    def __init__(self, datasets: List[Union[TimeSeries, Tuple[TimeSeries, TimeSeries], BenchmarkDataset]] = None,
                 models: List[BenchmarkModelHolder] = None,
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
        if not isinstance(model, Benchmark.BenchmarkModelHolder):
            raise TypeError(f"models must implement {Benchmark.BenchmarkModelHolder}, found{type(model)}.")
        self.models.append(model)

    def get_models(self):
        return [m.model_class for m in self.models]

    def add_dataset(self, dataset: Union[TimeSeries, Tuple[TimeSeries, TimeSeries], BenchmarkDataset], name=None):
        if not (isinstance(dataset, TimeSeries) or
                isinstance(dataset, Tuple) or
                isinstance(dataset, Benchmark.BenchmarkDataset)):
            raise TypeError(
                f"datasets must be of type {Benchmark.BenchmarkDataset}, {TimeSeries} or {(TimeSeries, TimeSeries)}, "
                f"found type {type(dataset)}")
        multivariate = True
        if (isinstance(dataset, Benchmark.BenchmarkDataset) and len(dataset.training_set.columns) < 2) \
                or (isinstance(dataset, TimeSeries) and len(dataset.columns) < 2) \
                or (isinstance(dataset, Tuple) and len(dataset[0].columns) < 2):
            multivariate = False

        if name is None:
            name = str(len(self.models))
        if isinstance(dataset, TimeSeries):
            self.datasets.append(Benchmark.BenchmarkDataset(dataset, multivariate, name))
        elif isinstance(dataset, Tuple):
            self.datasets.append(Benchmark.BenchmarkDataset(dataset[0], multivariate, name, dataset[1]))
        else:  # it's a BenchmarkDataset
            self.datasets.append(dataset)

    def get_datasets(self):
        return [d for d in self.datasets]

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
            # multivariate/univariate support measures
            nb_mv_run_failed = len([x for x in self.datasets if x.multivariate])
            nb_uv_run_failed = len(self.datasets) - nb_mv_run_failed
            mv0 = len([x for x in self.datasets if x.multivariate])

            # evaluation time!
            for dataset in self.datasets:
                if verbose:
                    print(f"on dataset {dataset.name} ")

                # initialize variables
                train_set, test_set = dataset.get_train_test_split(self.train_proportion)
                nb_features = len(train_set.columns)
                train_size = len(train_set.time_index)
                test_size = len(test_set.time_index)
                train_time = 0
                inference_time = 0
                # create new model
                source_model.instantiate(train_set=train_set, test_set=test_set)
                if verbose:
                    print(f"training... ", end="")
                train_success = True
                try:
                    # train
                    start_time = time.time()
                    source_model.fit(train_set, test_set)
                    train_time = time.time() - start_time

                    # test
                    if verbose:
                        print(f"testing... ", end="")
                    start_time = time.time()
                    pred = source_model.predict(test_size, test_set)
                    inference_time = time.time() - start_time
                    if verbose:
                        print(f"done, took {inference_time}")

                except:  # can't train or test on this dataset
                    train_success = False
                    if dataset.multivariate:
                        nb_mv_run_failed -= 1
                    else:
                        nb_uv_run_failed -= 1
                    if verbose:
                        print(f"Couldn't complete training.")
                        if debug:
                            traceback.print_exc()

                    if train_success:
                        # compute metrics
                        self.results[source_model.name][dataset.name] = {
                            'nb features': nb_features,
                            'training set size': train_size,
                            'training time': train_time,
                            'test set size': test_size,
                            'testing time': inference_time
                        }
                        # compute user-submitted metrics
                        for metric in self.metrics:
                            try:
                                if verbose:
                                    print(f"{metric.name}: ", end="")
                                self.results[source_model.name][dataset.name][metric.name] = metric.compute(test_set, pred)
                                if verbose:
                                    print(self.results[source_model.name][dataset.name][metric.name])
                            except: # can't compute current metric
                                print(f"Couldn't compute {metric}")
                                if debug:
                                    traceback.print_exc()

            # back to model-level stats
            supports_uni = None
            if len(self.datasets) - mv0 > 0:  # if we have univariate ds in our batch
                supports_uni = (nb_uv_run_failed > 0)
            supports_mult = None
            if mv0 > 0:
                supports_mult = (nb_mv_run_failed > 0)
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
                    s += f"{dataset.name}: couldn't complete training\n"
                else:
                    for key in self.results[model.name][dataset.name].keys():
                        s += f"{key}: {self.results[model.name][dataset.name][key]}\n"

            txt.append(s)
        report = "\n".join(txt)
        return report

    def get_report_dataframes(self) -> dict:
        # DISCLAIMER: this is still bugged
        # 1 dataframe per model
        # index = datasets
        # columns = metric
        ds_index = [x for x in self.results[self.models[0].name].keys() if
                    'supports' not in x]  # remove model-only stats
        me_columns = self.results[self.models[0].name][self.datasets[0].name].keys()
        reports = {}
        for model in self.models:
            report_model = pd.DataFrame(columns=me_columns, index=ds_index)
            for dataset in self.datasets:
                if dataset.name not in self.results[model.name].keys():
                    report_dict = {n: np.nan for n in me_columns}
                else:
                    report_dict = self.results[model.name][dataset.name]
                report_model.loc[dataset.name] = report_dict
            reports[model.name] = report_model
        return reports
