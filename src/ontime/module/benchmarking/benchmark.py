from ontime.module.data.datasets import DatasetLoader as dl
from ontime.core.time_series.time_series import TimeSeries
from ontime.core.model.model import Model
from ontime.core.model.abstract_model import AbstractModel
from darts.metrics import metrics
import pandas as pd
import time
import copy

class Benchmark:

    def __init__(self, datasets = None, models = None):
        self.datasets = []
        self.models = []
        if datasets is not None:
            i=1
            for d in datasets:
                self.add_dataset(d, str(i))
                i+=1
        if models is not None:
            i=1
            for m in models:
                self.add_model(m, str(i))
                i+=1
        self.results = None

    def add_model(self, model, name = None):
        if not isinstance(model, AbstractModel):
                    raise TypeError(f"models must implement {AbstractModel}")
        if name is None:
            name = str(len(self.models))
        self.models.append({
             'model': model,
             'name': name
             })

    def get_models(self):
        return [m['model'] for m in self.models]
    
    def add_dataset(self, dataset, name = None):
        if not isinstance(dataset, TimeSeries):
                    raise TypeError(f"datasets must be of type {TimeSeries}, found type {type(dataset)}")
        multivariate = True
        if len(dataset.columns) < 2:
             multivariate = False
        if name is None:
            name = str(len(self.models))
        self.datasets.append({
             'dataset': dataset, 
             'name': name, 
             'is_multivariate': multivariate
             })

    def get_datasets(self):
        return [d['dataset'] for d in self.datasets]
    
    def run(self, test_proportion = 0.3, verbose = False):
        #TODO: throw error if models or datasets is empty
        self.results = []
        if verbose: print("Starting evaluation...")
        i=0
        for sourcemodel in self.models:
            self.results.append([])
            if verbose: print(f"Evaluation for model {sourcemodel['name']}")
            resi = pd.DataFrame(index = ['nb features', 'train size', 'train time', 'test size', 'test time', 'mape'])
            nb_mv = len([x for x in self.datasets if x['is_multivariate']])
            nb_uv = len(self.datasets) - nb_mv
            mv0 = len([x for x in self.datasets if x['is_multivariate']])
            for dataset in self.datasets:
                if verbose: print(f"on dataset {dataset['name']} ")
                model = copy.deepcopy(sourcemodel['model'])
                train_set, test_set = dataset['dataset'].split_before(test_proportion)
                nb_features = len(dataset['dataset'].columns)
                train_size = len(train_set.time_index)
                test_size = len(test_set.time_index)
                #test multivariate
                #train 
                if verbose: print(f"train ", end="")
                #print(train_set)
                try:
                    start_time = time.time()
                    model.fit(train_set)
                    train_time = time.time() - start_time    
                    #test
                    if verbose: print(f"infer ", end="")
                    steps_to_predict = len(test_set.time_index.tolist())
                    start_time = time.time()
                    pred = model.predict(steps_to_predict)
                    inference_time = time.time() - start_time
                    if verbose: print(f"done, took {inference_time}")
                    mape = metrics.mape(test_set, pred)
                    resi[dataset['name']] = [nb_features, train_size, train_time, test_size, inference_time, mape]
                except: # can't train on this dataset
                    if dataset['is_multivariate']:
                        nb_mv -= 1
                    else:
                        nb_uv -= 1
                    if verbose: print(f"Couldn't complete training.")

            suni = None
            if len(self.datasets)-mv0 > 0: # if we have univariate ds in our batch
                 suni = (nb_uv > 0)
            smul = None
            if mv0>0:
                smul = (nb_mv > 0)
            self.results[i] = {
                 'metrics': resi,
                 'supports univariate': suni,
                 'supports multivariate': smul
            }
            i+=1
        #return self.results
    
    def get_report(self):
        if self.results is None:
            return "please invoke run_benchmark() to generate report data"
        txt = []
        
        for i in range(0,len(self.models)):
            suni = self.results[i]['supports univariate']
            smul = self.results[i]['supports multivariate']
            match suni:
                 case True: suni = '✓'
                 case False: suni = 'X'
                 case None: suni = "unknown"
            match smul:
                 case True: smul = '✓'
                 case False: smul = 'X'
                 case None: smul = "unknown"
            model = self.models[i]
            s = f"Model {model['name']}:\n"
            s += f"Supported univariate datasets: {suni}\n"
            s += f"Supported multivariate datasets: {smul}\n"
            s += f"{self.results[i]['metrics']}\n"

            txt.append(s)
        report = "\n".join(txt)
        return report
                    