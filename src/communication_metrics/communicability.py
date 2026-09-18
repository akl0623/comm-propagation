#!/usr/bin/env python3
"""Degree-normalized communicability (matrix exponential) metric wrapper.

Exposes communicability = degree-normalized matrix exponential:
    s_i = sum_j W_ij ;  D = diag(1/sqrt(s_i)) (s_i=1 protection for zero
    strength);  G = expm(D @ W @ D)  (scipy.linalg.expm, Pade approximation,
    diagonal preserved).
Implementation shared with `compute_comm_matrices.py` (function
`calculate_cmy`). This wrapper adds no logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_comm_matrices import calculate_cmy  # noqa: E402,F401

__all__ = ['calculate_cmy']
