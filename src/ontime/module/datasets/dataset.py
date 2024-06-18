from abc import abstractmethod

from darts import TimeSeries as dTimeSeries
from darts import datasets as dd
from sklearn.datasets import fetch_openml

from ...core.time_series import TimeSeries


class Dataset:
    """Dataset loader class for loading datasets from Darts, OpenML and other sources"""

    class ImportedDataset:
        """Class for dataset loaders"""

        def __init__(self, dataset, args={}):
            self.dataset = dataset
            self.args = args

        @abstractmethod
        def load(self) -> TimeSeries:
            pass

    class DartsDataset(ImportedDataset):
        """Darts dataset loader"""

        def load(self) -> TimeSeries:
            return TimeSeries.from_darts(self.dataset.load(*self.args))

    class OpenMLDataset(ImportedDataset):
        """OpenML dataset loader"""

        def load(self) -> TimeSeries:
            self.dataset.index = self.dataset[self.args["time_col"]]
            dts = dTimeSeries.from_dataframe(self.dataset, **self.args)
            return TimeSeries.from_darts(dts)

    # Darts datasets list
    AirPassengersDataset = DartsDataset(dd.AirPassengersDataset())
    AusBeerDataset = DartsDataset(dd.AusBeerDataset())
    AustralianTourismDataset = DartsDataset(dd.AustralianTourismDataset())
    ETTh1Dataset = DartsDataset(dd.ETTh1Dataset())
    ETTh2Dataset = DartsDataset(dd.ETTh2Dataset())
    ETTm1Dataset = DartsDataset(dd.ETTm1Dataset())
    ETTm2Dataset = DartsDataset(dd.ETTm2Dataset())
    # this one won't load for an unknown reason
    # ElectricityConsumptionZurichDataset = (dd.ElectricityConsumptionZurichDataset(), ''
    ElectricityDataset = DartsDataset(dd.ElectricityDataset())
    EnergyDataset = DartsDataset(dd.EnergyDataset())
    ExchangeRateDataset = DartsDataset(dd.ExchangeRateDataset())
    GasRateCO2Dataset = DartsDataset(dd.GasRateCO2Dataset())
    HeartRateDataset = DartsDataset(dd.HeartRateDataset())
    ILINetDataset = DartsDataset(dd.ILINetDataset())
    IceCreamHeaterDataset = DartsDataset(dd.IceCreamHeaterDataset())
    MonthlyMilkDataset = DartsDataset(dd.MonthlyMilkDataset())
    MonthlyMilkIncompleteDataset = DartsDataset(dd.MonthlyMilkIncompleteDataset())
    SunspotsDataset = DartsDataset(dd.SunspotsDataset())
    TaylorDataset = DartsDataset(dd.TaylorDataset())
    TemperatureDataset = DartsDataset(dd.TemperatureDataset())
    TrafficDataset = DartsDataset(dd.TrafficDataset())
    USGasolineDataset = DartsDataset(dd.USGasolineDataset())
    UberTLCDataset = DartsDataset(dd.UberTLCDataset())
    WeatherDataset = DartsDataset(dd.WeatherDataset())
    WineDataset = DartsDataset(dd.WineDataset())
    WoolyDataset = DartsDataset(dd.WoolyDataset())

    # OpenML datasets
    AMDStockPrices = OpenMLDataset(
        fetch_openml(
            "AMD-Stock-Prices-Historical-Data",
            version=1,
            as_frame=True,
            parser="pandas",
        ).frame,
        args={"time_col": "Date", "fill_missing_dates": True, "freq": "D"},
    )
