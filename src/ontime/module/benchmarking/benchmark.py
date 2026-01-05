from __future__ import annotations
from typing import List, Tuple, Any, Literal, Dict
import logging

from ontime import TimeSeries
from ontime.core.modelling.abstract_model import AbstractModel
from .benchmark_dataset import BenchmarkDataset
from .benchmark_evaluator import BenchmarkEvaluator
from .benchmark_metric import BenchmarkMetric
from .benchmark_model_config import BenchmarkModelConfig

from darts.dataprocessing.transformers import Scaler
from alive_progress import alive_bar
import pandas as pd
import time
import traceback
import json, pickle
import numpy as np
from tabulate import tabulate
import os

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger(
    name: str = None, logging_level: int = logging.WARNING
) -> logging.Logger:
    """
    Configures and returns a logger with the desired verbosity.

    :param name: The name of the logger. If None, defaults to "CustomLogger".
    :param logging_level: The logging level, e.g., logging.DEBUG, logging.INFO, logging.WARNING, etc.
                         Defaults to logging.WARNING.
    :return: A configured logger instance.
    """
    # Chat GPT generated
    logger = logging.getLogger(name or "BenchmarkLogger")
    logger.setLevel(logging_level)

    # Avoid adding handlers multiple times
    if not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging_level)
        logger.addHandler(handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    return logger


def save_data(
    data: Any,
    path: str,
    file_name: str,
    format: Literal["json", "pickle"] = "json",
) -> None:
    """
    Save a data object to a file in the specified format.

    :param data: The data object to save.
    :param path: The path where the file will be saved.
    :param file_name: The name of the file (without extension).
    :param format: The format to save the file in. Can be either "json" or "pickle", defaults to "json".
    :raises ValueError: If the specified type is not "json" or "pickle".
    """
    if format == "json":
        with open(f"{path}/{file_name}.json", "w") as f:
            json.dump(data, f, indent=4)
    elif format == "pickle":
        with open(f"{path}/{file_name}.pkl", "wb") as f:
            pickle.dump(data, f)
    else:
        raise ValueError("format must be either 'json' or 'pickle'")


class Benchmark:
    """
    Benchmark class to initialize a benchmark with models, datasets and metrics, run it, and retrieve results.
    """

    def __init__(
        self,
        model_configs: List[AbstractModel] = None,
        datasets: List[BenchmarkDataset] = None,
        metrics: List[BenchmarkMetric] = None,
        few_shot_proportions: List[float] = [1.0],
        result_dir: str = "benchmark_results",
    ):
        """
        Initializes a Benchmark

        :param model_configs: config of models to benchmark
        :param datasets: datasets on which to benchmark the models
        :param metrics: metrics used to benchmark the models
        :param few_shot_proportions: list of proportions of the train set to use for few-shot learning
        """
        self.datasets: List[BenchmarkDataset] = []
        self.model_configs: List[BenchmarkModelConfig] = []
        self.metrics: List[BenchmarkMetric] = []
        self.few_shot_proportions = few_shot_proportions

        # for holding results and predictions
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.results = {}
        self.dataset_info = {}
        self.predictions = {}

        # initialize datasets, models and metrics
        if metrics is not None:
            for m in metrics:
                self.add_metric(m)
        if datasets is not None:
            for d in datasets:
                self.add_dataset(d)
        if model_configs is not None:
            for m in model_configs:
                self.add_model_config(m)

    def add_model_config(self, model_config: BenchmarkModelConfig):
        """
        Add a model config to the benchmark

        :param model_config: the model configuration to add
        :raises TypeError: if the model class contained in given the model configuration is not a subclass of the ModelInterface interface
        """
        if not issubclass(model_config.model_class, AbstractModel):
            raise TypeError(
                f"models must implement {AbstractModel}, found{type(model_config)}."
            )
        self.model_configs.append(model_config)

    def add_dataset(self, dataset: BenchmarkDataset):
        """
        Add a dataset to the benchmark

        :param dataset: the dataset to add
        :raises TypeError: if the given dataset is not an instance of BenchmarkDataset class
        """
        if not isinstance(dataset, BenchmarkDataset):
            raise TypeError(
                f"datasets must be of type {BenchmarkDataset}, found type {type(dataset)}"
            )
        self.datasets.append(dataset)

    def add_metric(self, metric: BenchmarkMetric):
        """
        Add a metric to the benchmark

        :param metric: the metric to add
        :raises TypeError: if the given metric is not an instance of BenchmarkMetric class
        """
        self.metrics.append(metric)

    def run(
        self,
        logging_level: str = "warning",
        nb_predictions: int = 1,
        save_all_predictions: bool = False,
        run_name: str = None,
    ):
        """
        Run the benchmark

        :param logging_level: logging level, can either be debug, info, warning, error or critical. Default to warning
        :param nb_predictions: the number of predictions to do per model and dataset, for plotting purpose
        :param save_all_predictions: if True, save all predictions in the result directory, default to False
        :param run_name: name of the run, if None, a random name will be generated
        """
        logger = setup_logger(logging_level=LOG_LEVELS[logging_level])

        # if run_name is not None, create a random name for the run
        if run_name is None:
            run_name = f"benchmark_{int(time.time())}"
        run_dir = f"{self.result_dir}/{run_name}"
        os.makedirs(run_dir, exist_ok=True)
        if save_all_predictions:
            all_predictions_dir = f"{run_dir}/all_predictions"
            os.makedirs(all_predictions_dir, exist_ok=True)
        logger.info(f"Running benchmark {run_name}")

        total_steps = len(self.model_configs) * len(self.datasets)

        with alive_bar(
            total_steps, title="Benchmarking", force_tty=True, length=20, max_cols=200
        ) as bar:
            inputs, targets = self._get_random_inputs(nb_predictions)
            self.predictions = {"inputs": inputs, "targets": targets, "predictions": {}}

            for dataset in self.datasets:
                self.results[dataset.name] = dataset_results = {}
                self.predictions["predictions"][dataset.name] = dataset_predictions = {}

                logger.info(f"On {dataset.name} dataset...")

                nb_features = dataset.ts.n_components
                full_train_set, test_set = dataset.get_train_test_split()
                full_train_size = len(full_train_set.time_index)
                test_size = len(test_set.time_index)

                evaluator = BenchmarkEvaluator(dataset, self.metrics)

                self.dataset_info[dataset.name] = {
                    "nb features": nb_features,
                    "target column": dataset.target_columns,
                    "training set size": full_train_size,
                    "test set size": test_size,
                    "validation set proportion": dataset.validation_proportion,
                }

                for model_config in self.model_configs:
                    bar.text(f"{model_config.model_name} on {dataset.name}")
                    bar()

                    logger.info(f"On {model_config.model_name} model...")

                    dataset_results[model_config.model_name] = model_results = {}
                    dataset_predictions[model_config.model_name] = model_predictions = (
                        {}
                    )

                    if model_config.zero_shot_only:
                        few_shot_proportions = [0.0]
                    else:
                        few_shot_proportions = self.few_shot_proportions

                    model = model_config.init_model(dataset=dataset)

                    for few_shot_proportion in few_shot_proportions:
                        bar.text(
                            f"{model_config.model_name} on {dataset.name}, using {few_shot_proportion*100:.1f}% of training data"
                        )
                        model_results[few_shot_proportion] = results = {}
                        model_predictions[few_shot_proportion] = predictions = []

                        few_shot_train_set = full_train_set[
                            : int(full_train_size * few_shot_proportion)
                        ]
                        train_set, val_set = dataset.get_train_val_split(
                            few_shot_train_set
                        )

                        # scaling (only if scaler is not None and not zero-shot)
                        if (
                            dataset.scaler_type is not None
                            and few_shot_proportion > 0.0
                        ):
                            scaler = Scaler(dataset.scaler_type())
                            train_set = scaler.fit_transform(train_set)
                            val_set = scaler.transform(val_set)
                            test_set = scaler.transform(test_set)
                        else:
                            scaler = None

                        times = {}

                        try:
                            if few_shot_proportion > 0.0:
                                logging.info("Training ...")
                                start_time = time.time()
                                fit_kwargs = {
                                    **model_config.get_fit_kwargs(dataset),
                                    **{"ts": train_set},
                                }
                                if model_config.validation_set_param is not None:
                                    fit_kwargs[model_config.validation_set_param] = (
                                        val_set
                                    )
                                model.fit(**fit_kwargs)
                                times["training"] = time.time() - start_time
                                logger.info(
                                    f"Training done, it took {times['training']}"
                                )
                            else:
                                logging.info("Training skipped, zero-shot evaluation")
                                times["training"] = 0

                            logger.info("Evaluating...")

                            predict_kwargs = model_config.get_predict_kwargs(dataset)

                            start_time = time.time()
                            eval_results = evaluator.evaluate(
                                model=model,
                                scaler=scaler,
                                return_predictions=save_all_predictions,
                                batch_size=model_config.test_batch_size(dataset),
                                univariate_prediction=model_config.is_univariate,
                                predict_kwargs=predict_kwargs,
                            )
                            if save_all_predictions:
                                metrics, energy, all_predictions = eval_results
                                save_data(
                                    all_predictions,
                                    all_predictions_dir,
                                    f"{model_config.model_name}_{dataset.name}_{few_shot_proportion}",
                                    format="pickle",
                                )
                            else:
                                metrics, energy = eval_results

                            save_data(energy, f"{run_dir}", "energy", format="json")

                            times["evaluation"] = time.time() - start_time

                            logger.info(f"Evaluation done, took {times['evaluation']}")

                            # get predictions
                            if nb_predictions > 0:
                                logger.info(f"getting predictions... ")
                                predictions_time = []
                                for input in inputs[dataset.name]:
                                    start_time = time.time()
                                    prediction = model.predict(
                                        ts=input,
                                        n=dataset.target_length,
                                        **predict_kwargs,
                                    )
                                    predictions_time.append(time.time() - start_time)
                                    predictions.append(prediction)
                                times["inference"] = np.mean(predictions_time)
                        except:
                            results["suceeded"] = False
                            logger.warning(
                                f"Could not complete evaluation for {model_config.model_name}"
                                f" model on {dataset.name} dataset"
                            )
                            logger.debug(traceback.format_exc())

                        if not "suceeded" in results:
                            results.update(
                                {
                                    "suceeded": True,
                                    "times": times,
                                    "metrics": metrics,
                                }
                            )

                            logger.info(f"Computed metrics: \n {metrics}")

                        save_data(
                            self.predictions,
                            f"{run_dir}",
                            "predictions",
                            format="pickle",
                        )
                        save_data(self.results, f"{run_dir}", "results", format="json")

    def get_results(self):
        return self.results

    def get_predictions(self):
        return self.predictions

    def get_dataset_info(self):
        return self.dataset_info

    def get_report(self) -> str:
        """
        Generate a report in text format containing dataset information and model performances (both in time and metrics)

        :return: report in text format
        """
        if not self.results:
            return "please invoke run_benchmark() to generate report data"

        report = []

        for dataset, models in self.results.items():
            report.append(f"\nDataset: {dataset}\n")

            # Print dataset info dynamically
            if dataset in self.dataset_info:
                info = self.dataset_info[dataset]
                dataset_table = [
                    [key, value] for key, value in info.items()
                ]  # Extract keys/values dynamically
                report.append(tabulate(dataset_table, tablefmt="plain"))

            report.append("\nResults:\n")

            all_times_keys = set()  # Collect all possible time-related keys
            all_metrics_keys = set()  # Collect all possible metric keys

            # time and metric columns
            for model, proportions in models.items():
                for proportion, data in proportions.items():
                    all_times_keys.update(data.get("times", {}).keys())
                    all_metrics_keys.update(data.get("metrics", {}).keys())

            # for consistent sorting
            all_times_keys = sorted(all_times_keys)
            all_metrics_keys = sorted(all_metrics_keys)

            headers = (
                ["Model", "Few-shot %", "", "Success", ""]
                + all_times_keys
                + [""]
                + all_metrics_keys
            )
            table = []

            for model, proportions in models.items():
                for proportion, data in proportions.items():
                    times_values = [
                        f"{data['times'].get(k, 0):.2f}" for k in all_times_keys
                    ]

                    metrics_values = [
                        f"{data['metrics'].get(k, 0):.3f}" for k in all_metrics_keys
                    ]

                    row = (
                        [
                            model,
                            f"{proportion * 100:.0f}%",
                            "",
                            self._bool_to_symbol(data["suceeded"]),
                            "",
                        ]
                        + times_values
                        + [""]
                        + metrics_values
                    )
                    table.append(row)

            report.append(
                tabulate(table, headers=headers, tablefmt="grid")
            )  # generate table

        return "\n".join(report)

    @staticmethod
    def get_results_df(
        results: Dict, with_metrics: bool = True, with_times: bool = True
    ) -> pd.DataFrame:
        """
        Generate a dataframe from the benchmark results

        :param results: the benchmark results
        :param with_metrics: whether to include metrics in the benchmark results dataframe
        :param with_times: whether to include times in the benchmark results dataframe
        :return: the dataframe
        """
        flat_results = {}

        for dataset_name, models in results.items():
            for model_name, proportions in models.items():
                for few_shot_proportion, results in proportions.items():
                    proportion_str = f"{float(few_shot_proportion) * 100:.1f}%"

                    for key, values in results.items():
                        if key == "times" and with_times:
                            for time_name, time_value in values.items():
                                flat_results.setdefault(
                                    (model_name, proportion_str, time_name), {}
                                )[dataset_name] = time_value

                        elif key == "metrics" and with_metrics:
                            for metric_name, metric_value in values.items():
                                flat_results.setdefault(
                                    (model_name, proportion_str, metric_name), {}
                                )[dataset_name] = metric_value

        results_df = pd.DataFrame.from_dict(flat_results, orient="index")

        metric_time_index_name = (
            "Metric/Time"
            if with_metrics and with_times
            else "Metric" if with_metrics else "Time"
        )

        results_df.index.names = [
            "Model",
            "Few-shot proportion",
            metric_time_index_name,
        ]

        return results_df

    def get_report_dfs(
        self, with_metrics: bool = True, with_times: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate report as two dataframes, one for the dataset information, and one for the benchmark results

        :param with_metrics: whether to include metrics in the benchmark results dataframe
        :param with_times: whether to include times in the benchmark results dataframe
        :return: the two dataframes
        """

        if not self.results:
            return "please invoke run_benchmark() to generate report data"

        results_df = self.get_results_df(
            self.results, with_metrics=with_metrics, with_times=with_times
        )

        metric_time_index_name = (
            "Metric/Time"
            if with_metrics and with_times
            else "Metric" if with_metrics else "Time"
        )

        results_df.index.names = [
            "Model",
            "Few-shot proportion",
            metric_time_index_name,
        ]

        ds_info_df = pd.DataFrame.from_dict(self.dataset_info, orient="index").T
        ds_info_df.index.name = "Characteristic"

        return ds_info_df, results_df

    @staticmethod
    def _bool_to_symbol(b: bool) -> str:
        """Return a string representation of a boolean value, as a symbol

        Symbols are "✓" for True and "X" for False. If the input is None, returns "unknown"

        Args:
            b: bool value to be converted

        Returns:
            str: string representation of the boolean value as a symbol
        """
        if b is None:
            return "unknown"
        if b:
            return "✓"
        return "X"

    def _get_random_inputs(
        self, nb_samples: int = 1
    ) -> Tuple[dict[str, list[TimeSeries]], dict[str, list[TimeSeries]]]:
        """
        Retrieve ``nb_samples`` tuples of input and target from all dataset in the benchmark

        :param nb_samples: number of input/target tuples to retrieve
        :return: a dictionnary identified by dataset name, containing the tuples of input and target
        """

        # for each dataset, get the number of inputs specified, randomly selected.
        inputs = {}
        targets = {}

        for dataset in self.datasets:
            inputs[dataset.name] = []
            targets[dataset.name] = []
            if nb_samples < 1:
                continue
            _, test_set = dataset.get_train_test_split()

            # store dataset attributes
            input_length = dataset.input_length
            horizon = dataset.target_length
            stride = dataset.stride
            gap = dataset.gap

            # select random indices
            window_length = input_length + horizon + gap
            max_idx = len(test_set) - window_length
            available_indices = list(range(0, max_idx, stride))

            if len(available_indices) < nb_samples:
                nb_samples = len(available_indices)

            indices = np.random.choice(available_indices, nb_samples, replace=False)

            # store input and target time series
            for idx in indices:
                inputs[dataset.name].append(test_set[idx : idx + input_length])
                targets[dataset.name].append(
                    test_set[
                        idx + input_length + gap : idx + input_length + gap + horizon
                    ]
                )

        return inputs, targets
