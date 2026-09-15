"""Validation tools for active Beris-Edwards simulations.

The supported user API is dimension-explicit: use
:mod:`activebe_validator.two_d` or :mod:`activebe_validator.three_d`.
"""

from . import three_d, two_d
from ._version import __version__

__all__ = ["__version__", "three_d", "two_d"]

