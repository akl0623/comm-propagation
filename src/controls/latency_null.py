#!/usr/bin/env python3
"""Latency-permutation null control wrapper.

Exposes the latency-permutation null control (paper: LatencyNull).
Implementation is provided in `screen_controls.py` (function
`control_latency_perm`): onset latencies permuted within the seed's
hemisphere; full path search + B5 screen re-run on the SAME C4/C6/C8R
pipeline. This wrapper adds no logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screen_controls import control_latency_perm  # noqa: E402,F401

__all__ = ['control_latency_perm']
