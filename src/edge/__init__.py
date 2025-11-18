"""
Edge-based diagnostics and PDE tools.

This package groups the original SEMCOG OD edge-line modules
so that node-level code can evolve independently.
"""

from .. import adjacency  # re-export for convenience
from .. import cycles
from .. import gravity
from .. import hodge
from .. import locality
from .. import pde_fit
from .. import export_figs

__all__ = [
    "adjacency",
    "cycles",
    "gravity",
    "hodge",
    "locality",
    "pde_fit",
    "export_figs",
]

