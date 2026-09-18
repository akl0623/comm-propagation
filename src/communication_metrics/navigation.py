#!/usr/bin/env python3
"""Navigation efficiency metric wrapper.

Exposes the navigation efficiency metric. The implementation is shared
with `compute_comm_matrices.py` (function `navigation_efficiency`).
This wrapper adds no logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_comm_matrices import navigation_efficiency  # noqa: E402,F401

__all__ = ['navigation_efficiency']
