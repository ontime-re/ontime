from __future__ import annotations
from typing import List, Tuple
import logging

from ontime import TimeSeries
from ontime.core.modelling.model_interface import ModelInterface
from ontime.module.benchmarking import (
    BenchmarkDataset,
    BenchmarkModelConfig,
    BenchmarkMetric,
    BenchmarkMode,
    BenchmarkEvaluator
)

from alive_progress import alive_bar
import pandas as pd
import time
import traceback
import numpy as np

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

def setup_logger(name: str = None, logging_level: int = logging.WARNING) -> logging.Logger:
    """
    Configures and returns a logger with the desired verbosity.

    :param name: The name of the logger. If None, defaults to "CustomLogger".
    :param logging_level: The logging level, e.g., logging.DEBUG, logging.INFO, logging.WARNING, etc.
                         Defaults to logging.WARNING.
    :return: A configured logger instance.
    """
    logger = logging.getLogger(name or "CustomLogger")
    logger.setLevel(logging_level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging_level)
        logger.addHandler(handler)

    return logger

class Benchmark:
    """
    Benchmark class to initialize a benchmark with models, datasets and metrics, run it, and retrieve results.
    """
    def __init__(
        self,
        model_configs: List[ModelInterface] = None,
        datasets: List[BenchmarkDataset] = None,
        metrics: List[BenchmarkMetric] = None,
    ):
        """
        Initializes a Benchmark

        :param model_configs: config of models to benchmark
        :param datasets: datasets on which to benchmark the models
        :param metrics: metrics used to benchmark the models
        """
        self.datasets: List[BenchmarkDataset] = []
        self.model_configs: List[BenchmarkModelConfig] = []
        self.metrics: List[BenchmarkMetric] = []

        # for holding results and predictions
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
        if not issubclass(model_config.model_class, ModelInterface):
            raise TypeError(
                f"models must implement {ModelInterface}, found{type(model_config)}."
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

    def run(self, logging_level: str = "warning", nb_predictions: int = 1): 
        """
        Run the benchmark

        :param logging_level: logging level, can either be debug, info, warning, error or critical. Default to warning
        :param nb_predictions: the number of predictions to do per model and dataset, for plotting purpose
        """
        logger = setup_logger(logging_level=LOG_LEVELS[logging_level])
        
        total_steps = len(self.model_configs) * len(self.datasets)
        
        with alive_bar(total_steps, title="Benchmarking", force_tty=True, length=20, max_cols=200) as bar:
            inputs, targets = self._get_random_inputs(nb_predictions)
            self.predictions = {"inputs": inputs, "targets": targets, "predictions": {}}
            
            for dataset in self.datasets:
                self.results[dataset.name] = {}
                self.model_stats[dataset.name] = {}
                self.predictions["predictions"][dataset.name] = {}
                
                logger.info(f"On {dataset.name} dataset...")
                
                nb_features = dataset.ts.n_components
                _, test_set = dataset.get_train_test_split()
                train_set, val_set = dataset.get_train_val_split()
                train_size = len(train_set.time_index)
                val_size = len(val_set.time_index)
                test_size = len(test_set.time_index)
                
                evaluator = BenchmarkEvaluator(dataset, self.metrics)
                
                self.dataset_info[dataset.name] = {
                    "nb features": nb_features,
                    "target column": dataset.target_columns,
                    "training set size": train_size,
                    "validation set size": val_size,
                    "test set size": test_size,
                }
                
                training_time = 0
                inference_time = 0
                
                for model_config in self.model_configs:
                    bar.text(f"{model_config.model_name} on {dataset.name}")
                    self.results[dataset.name][model_config.model_name] = {}
                    self.model_stats[dataset.name][model_config.model_name] = {}
                    
                    logger.info(f"{model_config.model_name} model...")

                    try:
                        model = model_config.init_model()
                        
                        if model_config.benchmark_mode != BenchmarkMode.ZERO_SHOT:
                            logging.info("Training ...")
                            start_time = time.time()
                            fit_kwargs = {"ts": train_set}
                            if model_config.validation_set_param is not None:
                                fit_kwargs[model_config.validation_set_param] = val_set
                            model.fit(**fit_kwargs)
                            training_time = time.time() - start_time
                            logger.info(f"Training done, it took {training_time} seconds")

                        logger.info("Evaluating...")
                            
                        start_time = time.time()
                        metrics = evaluator.evaluate(model=model)
                        evaluation_time = time.time() - start_time
                        
                        print(f"Evaluation done, took {evaluation_time}")
                            
                        inference_time = np.nan
                        # get predictions
                        if nb_predictions > 0:
                            self.predictions["predictions"][dataset.name][model_config.model_name] = []
                            print(f"getting predictions... ", end="")
                            predictions_time = []
                            for input in inputs[dataset.name]:
                                start_time = time.time()
                                prediction = model_config.predict(ts=input, n=dataset.horizon)
                                predictions_time.append(time.time() - start_time)
                                self.predictions["predictions"][dataset.name][model_config.model_name].append(prediction)
                            inference_time = np.mean(predictions_time)

                    except:
                        self.results[model_config.model_name][dataset.name] = {"suceeded": False}
                        logger.info(f"Could not complete evaluation for {model_config.model_name}" 
                                    "model on {dataset.name} dataset")
                        logger.debug(traceback.print_exc())
                            
                    if not "suceeded" in self.results[model_config.model_name][dataset.name]:                          
                        self.results[model_config.model_name][dataset.name] = {
                            "suceeded": True,
                            "training time": training_time,
                            "evaluation time": evaluation_time,
                            "inference time": inference_time,
                            "metrics": metrics
                        }
                            
                        logger.info(f"Computed metrics: \n {metrics}")
                        
    def get_results(self):
        return self.results

    def get_model_stats(self):
        return self.model_stats
    
    def get_predictions(self):
        return self.predictions

    def get_report(self) -> str:
        """
        Generate a report in text format containing dataset information and model performances (both in time and metrics)

        :return: report in text format
        """
        if self.results is None:
            return "please invoke run_benchmark() to generate report data"
        txt = []

        for dataset in self.datasets:
            s = f"{dataset.name} dataset:\n"
            for key, value in self.dataset_info[dataset.name].items():
                s += f"{key}: {value}"

            for model_config in self.model_configs:
                model_name = model_config.model_name
                s += f"{model_name} model:\n"
                for key, value in self.results[dataset.name][model_name].items():
                    if key == "suceeded":
                        s += f"{key}: {self._bool_to_symbol(value)}"
                    else:
                        s += f"{key}: {value}\n"
            txt.append(s)
        report = "\n\n".join(txt)
        return report

    def get_report_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate report as two dataframes, one for the dataset information, and one for the benchmark results

        :return: the two dataframes
        """
        if self.results is None:
            return "please invoke run_benchmark() to generate report data"
        flat_results = {}
        for dataset_name, models_results in self.results.items():
            for model_name, model_results in models_results.items():
                for result_key, result_value in model_results.items():
                    if result_key == "metrics":
                        for metric_name, metric_value in result_value.items():
                            flat_results[
                                (dataset_name, metric_name)
                            ] = flat_results.get((dataset_name, metric_name), {})
                            flat_results[(dataset_name, metric_name)][
                                model_name
                            ] = metric_value
                    elif result_key in ["training time", "evaluation time", "inference time"]:
                        flat_results[(dataset_name, result_key)] = flat_results.get(
                            (dataset_name, result_key), {}
                        )
                        flat_results[(dataset_name, result_key)][
                            model_name
                        ] = result_value

        results_df = pd.DataFrame.from_dict(flat_results, orient="index")
        results_df.index.names = ["Dataset", "Metric"]
        model_stats_df = pd.DataFrame(self.model_stats)
        model_stats_df.index.name = "Statistic"

        return model_stats_df, results_df    
    
    @staticmethod
    def _bool_to_symbol(b: bool) -> str:
        if b is None:
            return "unknown"
        if b:
            return "✓"
        return "X"
    
    def _get_random_inputs(self, nb_samples: int = 1) -> Tuple[dict[str, list[TimeSeries]], dict[str, list[TimeSeries]]]:
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
            horizon = dataset.horizon
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
                inputs[dataset.name].append(test_set[idx:idx+input_length])
                targets[dataset.name].append(test_set[idx+input_length+gap:idx+input_length+gap+horizon])
                        
        return inputs, targets
