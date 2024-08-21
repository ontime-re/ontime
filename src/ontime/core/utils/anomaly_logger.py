import pandas as pd
from ontime.core.time_series.binary_time_series import BinaryTimeSeries


class BinaryAnomalyLogger:
    """
    Class to log anomalies in a binary time series.
    """

    def __init__(self, description: str, max_entries: int = 1000):
        """
        Constructor for BinaryAnomalyLogger

        :param description: Description of the log
        :param max_entries: Max number of entries allowed in the log
        """
        self.description = description
        self.max_entries = max_entries
        self.log = pd.DataFrame(columns=["Timestamp", "Description", "Value"])

    def reset_log(self):
        """
        Reset the log
        """
        self.log = pd.DataFrame(columns=["Timestamp", "Description", "Value"])

    def log_anomalies(self, ts: BinaryTimeSeries) -> pd.DataFrame:
        """
        Log the BinaryTimeSeries

        :param ts: TimeSeries object
        :return Pandas DataFrame
        """
        # Create the current log
        df = ts.pd_dataframe()
        df = df.reset_index()
        print(df.columns)
        log_df = pd.DataFrame(
            {
                "Timestamp": df.iloc[:, 0],
                "Description": self.description,
                "Value": df.iloc[:, 1] == 1,
            }
        )
        # Merge w/ existing log
        self.log = pd.concat([self.log, log_df])
        # Sort by timestamp
        self.log.sort_values(by="Timestamp", inplace=True)
        # Remove duplicates
        self.log.drop_duplicates(subset="Timestamp", keep="last", inplace=True)
        # Trim the log
        if self.max_entries:
            self.log = self.log.tail(self.max_entries)
