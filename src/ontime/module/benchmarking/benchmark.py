from ontime.module.data.datasets import DatasetLoader as dl
from ontime.core.time_series.time_series import TimeSeries
from ontime.core.model.model import Model
from ontime.core.model.abstract_model import AbstractModel
import time
import copy

class Benchmark:
    def __init__(self, datasets = None, models = None):
        self.datasets = None
        self.models = None
        if datasets is not None:
            i=1
            self.datasets = []
            for d in datasets:
                if not isinstance(d, TimeSeries):
                    raise TypeError(f"datasets must be of type {TimeSeries}, found type {type(d)}")
                self.datasets.append((d, str(i)))
                i+=1
        if models is not None:
            i=1
            self.models = []
            for m in models:
                if not isinstance(m, AbstractModel):
                    raise TypeError(f"models must implement {AbstractModel}, found {type(m)}")
                self.models.append((m, str(i)))
                i+=1
        self.results = None

    @staticmethod
    def _mape(y, y_hat):
        n = len(y)
        sum = 0
        for i in range(0, n):
            sum += abs(y_hat[i] - y[i])/y[i]
        return sum * 100/n

    def add_model(self, model, name = None):
        if not isinstance(model, AbstractModel):
                    raise TypeError(f"models must implement {AbstractModel}")
        if name is None:
            name = str(len(self.models))
        self.models.append((model,name))

    def get_models(self):
        return [m[0] for m in self.models]
    
    def add_dataset(self, dataset, name = None):
        if not isinstance(dataset, TimeSeries):
                    raise TypeError(f"datasets must be of type {TimeSeries}, found type {type(dataset)}")
        if name is None:
            name = str(len(self.models))
        self.datasets.append((dataset, name))

    def get_datasets(self):
        return [d[0] for d in self.datasets]
    
    def run(self, test_proportion = 0.3, verbose = False):
        #TODO: throw error if models or datasets is empty
        self.results = []
        if verbose: print("Starting evaluation...")
        i=0
        for sourcemodel in self.models:
            self.results.append([])
            if verbose: print(f"Evaluation for model {sourcemodel[1]}")
            j=0
            for dataset in self.datasets:
                if verbose: print(f"on dataset {dataset[1]} ")
                j+=1
                model = copy.deepcopy(sourcemodel[0])
                train_set, test_set = TimeSeries.train_test_split(dataset[0], test_proportion)

                #train 
                if verbose: print(f"train ", end="")
                #print(train_set)
                start_time = time.time()
                model.fit(train_set)
                train_time = time.time() - start_time
                if verbose: print(f"done, took {train_time}")

                #test
                if verbose: print(f"infer ", end="")
                steps_to_predict = len(test_set.time_index.tolist())
                start_time = time.time()
                pred = model.predict(steps_to_predict)
                inference_time = time.time() - start_time
                if verbose: print(f"done, took {inference_time}")

                error = Benchmark._mape(test_set, pred)
                self.results[i].append([train_time, inference_time, error])
            i+=1
        return self.results
    
    def get_report(self):
        if self.results is None:
            return "please invoke run_benchmark() to generate report data"
        txt = []
        for i in range(0,len(self.models)):
            model = self.models[i]
            s = f"Model {model[1]}:\n"
            for j in range(0,len(self.datasets)):
                dataset = self.datasets[j]
                traintime = self.results[i][j][0]
                testtime = self.results[i][j][1]
                mape = self.results[i][j][2]
                mape = mape.pd_dataframe().iat[0,0]
                s += f"dataset {dataset[1]}\n"
                s += f"Results:\n"
                s += f"training time: {traintime}\n"
                s += f"test time: {testtime}\n"
                s += f"MAPE:  {mape}\n"
            txt.append(s)
        report = "\n".join(txt)
        return report
                    