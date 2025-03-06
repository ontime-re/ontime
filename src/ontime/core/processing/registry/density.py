


class Density(AbstractProcessor):
    """Density class handles density computation in a TimeSeries"""

    def __init__(self, window_length: int):
        """Constructor of a density processor

        :param window_length: int
        """
        self.window_length = window_length


    def process(self, ts: Union[UnitTimeSeries, BinaryTimeSeries], mode: str = 'absolute') -> Union[UnitTimeSeries, BinaryTimeSeries]:
        """Compute densities for a TimeSeries

        Two modes are available:
        - 'absolute': the density is the absolute number of anomalies in the window
        - 'relative': the density is the relative number of anomalies in the window between 0 and 1

        :param ts: TimeSeries
        :param mode: str
        :return: TimeSeries
        """
        assert isinstance(self.window_length, int), f"window_length must be an integer, not {type(self.window_length)}"
    
        match mode:
            case 'absolute':
                def count(x):
                    return np.sum(x)
            case 'relative':
                def count(x):
                    return np.sum(x)/window_length

        density_ts = on.TimeSeries.from_darts(
                    ts.window_transform(transforms={
                        'function': lambda x: count(x),
                        'mode': 'rolling',
                        'window': window_length,
                        'function_name': f'count_{mode}'
                    })
                )
        
        return density_ts
