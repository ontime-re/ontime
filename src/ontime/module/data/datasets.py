from darts import datasets as dd
from ...core.time_series import TimeSeries

class DatasetLoader():
    AirPassengersDataset = (dd.AirPassengersDataset(), 'darts')
    AusBeerDataset = (dd.AusBeerDataset(), 'darts')
    AustralianTourismDataset = (dd.AustralianTourismDataset(), 'darts')
    ETTh1Dataset = (dd.ETTh1Dataset(), 'darts')
    ETTh2Dataset = (dd.ETTh2Dataset(), 'darts')
    ETTm1Dataset = (dd.ETTm1Dataset(), 'darts')
    ETTm2Dataset = (dd.ETTm2Dataset(), 'darts')
    # this one won't load for an unknown reason
    # ElectricityConsumptionZurichDataset = (dd.ElectricityConsumptionZurichDataset(), 'darts')
    ElectricityDataset = (dd.ElectricityDataset(), 'darts')
    EnergyDataset = (dd.EnergyDataset(), 'darts')
    ExchangeRateDataset = (dd.ExchangeRateDataset(), 'darts')
    GasRateCO2Dataset = (dd.GasRateCO2Dataset(), 'darts')
    HeartRateDataset = (dd.HeartRateDataset(), 'darts')
    ILINetDataset = (dd.ILINetDataset(), 'darts')
    IceCreamHeaterDataset = (dd.IceCreamHeaterDataset(), 'darts')
    MonthlyMilkDataset = (dd.MonthlyMilkDataset(), 'darts')
    MonthlyMilkIncompleteDataset = (dd.MonthlyMilkIncompleteDataset(), 'darts')
    SunspotsDataset = (dd.SunspotsDataset(), 'darts')
    TaylorDataset = (dd.TaylorDataset(), 'darts')
    TemperatureDataset = (dd.TemperatureDataset(), 'darts')
    TrafficDataset = (dd.TrafficDataset(), 'darts')
    USGasolineDataset = (dd.USGasolineDataset(), 'darts')
    UberTLCDataset = (dd.UberTLCDataset(), 'darts')
    WeatherDataset = (dd.WeatherDataset(), 'darts')
    WineDataset = (dd.WineDataset(), 'darts')
    WoolyDataset = (dd.WoolyDataset(), 'darts')
    # add datasets from other sources here
    
    @staticmethod
    def load(dataset, *args):
        if dataset[1] == 'darts': #if from Darts
            return TimeSeries.from_darts(dataset[0].load(*args))
        #elif dataset[1] == 'xx':
            #load dataset + convert to TimeSeries ...