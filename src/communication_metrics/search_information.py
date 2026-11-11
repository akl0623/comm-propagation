#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_comm_matrices import search_information_fixed  # noqa: E402,F401

__all__ = ['search_information_fixed']
