#!/usr/bin/env python3
"""Search information metric wrapper.

Exposes search information (Goni et al. 2014 semantics). The implementation
is shared with `compute_comm_matrices.py` (function
`search_information_fixed`, an exact port of bct 0.6.0 search_information
with the numpy-2 scalar fix). This wrapper adds no logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_comm_matrices import search_information_fixed  # noqa: E402,F401

__all__ = ['search_information_fixed']
