#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_comm_matrices import calculate_cmy  # noqa: E402,F401

__all__ = ['calculate_cmy']
