"""
flasheda.engine
~~~~~~~~~~~~~~~
Size-aware execution layer: inspects a source cheaply, samples it
without a full load, and picks the right dataframe engine for the
full-size work.
"""

from .backends import ENGINES, available_engines, choose_engine, is_available
from .sampler import get_sample_size, sample, set_sample_size
from .source import SourceInfo, inspect_source

__all__ = [
    "ENGINES",
    "SourceInfo",
    "available_engines",
    "choose_engine",
    "get_sample_size",
    "inspect_source",
    "is_available",
    "sample",
    "set_sample_size",
]
