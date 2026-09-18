#!/usr/bin/env python3
"""Matched-random negative control wrapper.

Exposes the matched-random negative control. Implementation is provided in
`screen_controls.py` (function `control_matched_random`): random SC walks
(latency ignored, no repeats) matched on L (same cell), hemisphere,
seed/endpoint degree and total tract length (+-25% of the real per-cell
median); features computed directly. Runs the SAME C4/C6/C8R pipeline as
the main experiment. This wrapper adds no logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screen_controls import control_matched_random  # noqa: E402,F401

__all__ = ['control_matched_random']
