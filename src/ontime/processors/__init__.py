from .processors import Processors
from .registry.filler import Filler
from .registry.mapper import Mapper
from .registry.windower import Windower
from .registry.correlation import Correlation

processors = Processors()
processors.load("filler", Filler)
processors.load("mapper", Mapper)
processors.load("windower", Windower)
processors.load("correlation", Correlation)
