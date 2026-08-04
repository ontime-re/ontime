from .processors import Processors
from .registry.filler import Filler
from .registry.mapper import Mapper
from .registry.windower import Windower
from .registry.correlation import Correlation
from .registry.density import Density
from .registry.fourier import Fourier

processors = Processors()
processors.load("filler", Filler)
processors.load("mapper", Mapper)
processors.load("windower", Windower)
processors.load("correlation", Correlation)
processors.load("density", Density)
processors.load("fourier", Fourier)

__all__ = ["processors"]
