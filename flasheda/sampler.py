"""
flasheda.sampler
~~~~~~~~~~~~~~~~
Kept for backward compatibility — sampling now lives in flasheda.engine.
"""

from .engine.sampler import get_sample_size, sample, set_sample_size

__all__ = ["sample", "get_sample_size", "set_sample_size"]
