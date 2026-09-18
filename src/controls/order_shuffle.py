#!/usr/bin/env python3
"""Order-shuffle negative control wrapper.

Exposes the order-shuffle negative control. Implementation is provided in
`screen_controls.py` (function `control_order_shuffle`): intermediates of
each real screened path are shuffled (SC-edge validity + metric finiteness
enforced; temporal check deliberately not applied - the control is defined
to break temporal order). Runs the SAME C4/C6/C8R pipeline as the main
experiment. This wrapper adds no logic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screen_controls import control_order_shuffle  # noqa: E402,F401

__all__ = ['control_order_shuffle']
