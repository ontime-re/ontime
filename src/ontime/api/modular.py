"""
onTime Modular API Definition

The aim of the Modular API is to give building blocks to the user to build whatever is desired.
`module` and `context` are left as is.

The core of the API is accessible through with the main object of the library `onTime`.
For instance :

    import ontime as on
    on.TimeSeries()

Features contained in `module` and `context` are accessible through the `onTime` object.
For instance :

    import ontime.module as onm
    onm.processing.common.my_function()

"""

from ..core import (
    detectors,
    generators,
    Model,
    models,
    Plot,
    marks,
    processors,
    TimeSeries,
)

from .. import module
from .. import context

__all__ = [
    # core
    "detectors",
    "generators",
    "Model",
    "models",
    "Plot",
    "marks",
    "processors",
    "TimeSeries",
    # module
    "module",
    # context
    "context",
]
