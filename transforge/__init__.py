"""TransForge: a verified optimal-transport solver bench.

Layering contract (enforced by tests/test_architecture.py):

    cli -> pipeline -> {data, solvers, adapt, eval} -> core

``core`` never imports any other project module.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星 (CJX0712)"
__all__ = ["__version__", "__author__"]
