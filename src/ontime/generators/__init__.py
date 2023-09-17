from .generators import Generators
from .registry.constant import Constant
from .registry.gaussian import Gaussian
from .registry.holiday import Holiday
from .registry.linear import Linear
from .registry.random_walk import RandomWalk
from .registry.sine import Sine

generators = Generators()
generators.load("constant", Constant)
generators.load("gaussian", Gaussian)
generators.load("holiday", Holiday)
generators.load("linear", Linear)
generators.load("random_walk", RandomWalk)
generators.load("sine", Sine)
