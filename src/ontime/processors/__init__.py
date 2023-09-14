from .processors import Processors
from .registry.filler import Filler
from .registry.mapper import Mapper
from .registry.windower import Windower

processors = Processors()
processors.load("filler", Filler)
processors.load("mapper", Mapper)
processors.load("windower", Windower)
