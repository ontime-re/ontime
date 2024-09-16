from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from enum import Enum

from ontime.core.time_series.time_series import TimeSeries
from ontime.module.benchmarking import (
    BenchmarkDataset,
    AbstractBenchmarkModel,
    BenchmarkMetric,
    BenchmarkMode,
)

import pandas as pd
import time
import traceback


class Benchmark:
    def __init__(
        self,
        datasets: List[BenchmarkDataset] = None,
        models: List[AbstractBenchmarkModel] = None,
        metrics: List[BenchmarkMetric] = None,
    ):
        self.datasets = []
        self.models = []
        self.metrics = []

        # for holding results
        self.metrics_results = []
        self.model_stats = {}
        self.results = {}

        # initialize datasets, models and metrics
        if metrics is not None:
            self.metrics = metrics
        if datasets is not None:
            i = 1
            for d in datasets:
                self.add_dataset(d)
                i += 1
        if models is not None:
            for m in models:
                self.add_model(m)

    def add_model(self, model):
        if not isinstance(model, AbstractBenchmarkModel):
            raise TypeError(
                f"models must implement {AbstractBenchmarkModel}, found{type(model)}."
            )
        self.models.append(model)

    def get_models(self):
        return [m.model_class for m in self.models]

    def add_dataset(self, dataset: BenchmarkDataset):
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError(
                f"datasets must be of type {BenchmarkDataset}, found type {type(dataset)}"
            )
        self.datasets.append(dataset)

    def get_datasets(self):
        return [d for d in self.datasets]

    def add_metric(self, metric: BenchmarkMetric):
        self.metrics.append(metric)

    def get_metrics(self):
        return self.metrics

    def run(self, verbose: bool = False, debug: bool = False):
        # TODO: throw error if models or datasets is empty
        if verbose:
            print("Starting evaluation...")
        i = 0
        for source_model in self.models:
            self.results[source_model.name] = {}
            self.model_stats[source_model.name] = {}
            if verbose:
                print(f"Evaluation for model {source_model.name}")

            # internal metrics (computed during run time)
            mv0 = len([x for x in self.datasets if x.is_multivariate()])
            nb_mv_run_succeeded = mv0
            nb_uv_run_succeeded = len(self.datasets) - mv0

            # evaluation time!
            for dataset in self.datasets:
                print(f"on dataset {dataset.name}")

                # initialize variables
                nb_features = dataset.ts.n_components
                train_set, test_set = dataset.get_train_test_split()
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
                    metrics = source_model.evaluate(
                        test_set,
                        dataset.horizon,
                        self.metrics,
                        context_length=dataset.input_length,
                        stride_length=dataset.stride,
                    )
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
                        "nb features": nb_features,
                        "target column": dataset.target_columns,
                        "training set size": train_size,
                        "training time": train_time,
                        "test set size": test_size,
                        "testing time": inference_time,
                    }
                    # compute user-submitted metrics
                    if verbose:
                        print(f"Computed metrics: {metrics}")

                    self.results[source_model.name][dataset.name]["metrics"] = metrics

            # back to model-level stats
            supports_uni = None
            if len(self.datasets) - mv0 > 0:  # if we have univariate ds in our batch
                supports_uni = nb_uv_run_succeeded > 0
            supports_mult = None
            if mv0 > 0:
                supports_mult = nb_mv_run_succeeded > 0
            self.model_stats[source_model.name][
                "supports univariate"
            ] = Benchmark._bool_to_symbol(supports_uni)
            self.model_stats[source_model.name][
                "supports multivariate"
            ] = Benchmark._bool_to_symbol(supports_mult)
            i += 1

    @staticmethod
    def _bool_to_symbol(b: bool) -> str:
        if b is None:
            return "unknown"
        if b:
            return "✓"
        return "X"

    def get_report(self):
        if self.results is None:
            return "please invoke run_benchmark() to generate report data"
        txt = []

        for model in self.models:
            s = f"Model {model.name}:\n"
            s += f"Supported univariate datasets: {self.model_stats[model.name]['supports univariate']}\n"
            s += f"Supported multivariate datasets: {self.model_stats[model.name]['supports multivariate']}\n"
            for dataset in self.datasets:
                s += f"Dataset {dataset.name}:\n"
                if dataset.name not in self.results[model.name].keys():
                    s += f"dataset.name couldn't complete training\n"
                else:
                    for key in self.results[model.name][dataset.name].keys():
                        if key == "prediction":
                            continue
                        s += f"{key}: {self.results[model.name][dataset.name][key]}\n"
            txt.append(s)
        report = "\n\n".join(txt)
        return report

    def get_report_df(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        flat_results = {}
        for model_name, datasets in self.results.items():
            for dataset_name, dataset_results in datasets.items():
                for result_name, result_value in dataset_results.items():
                    if result_name == "metrics":
                        for metric_name, metric_value in result_value.items():
                            flat_results[
                                (dataset_name, metric_name)
                            ] = flat_results.get((dataset_name, metric_name), {})
                            flat_results[(dataset_name, metric_name)][
                                model_name
                            ] = metric_value
                    elif result_name in ["training time", "testing time"]:
                        flat_results[(dataset_name, result_name)] = flat_results.get(
                            (dataset_name, result_name), {}
                        )
                        flat_results[(dataset_name, result_name)][
                            model_name
                        ] = result_value

        results_df = pd.DataFrame.from_dict(flat_results, orient="index")
        results_df.index.names = ["Dataset", "Metric"]
        model_stats_df = pd.DataFrame(self.model_stats)
        model_stats_df.index.name = "Statistic"

        return model_stats_df, results_df

    def get_results(self):
        return self.results

    def get_model_stats(self):
        return self.model_stats
