#!/usr/bin/env python3
"""Routing efficiency metric wrapper.

Routing efficiency = bct.rout_efficiency(L, transform=None)[1] (BCT 0.6.0
semantics, shortest-path length matrix input). This is the identical call
used inside `communication_metrics.py::compute_all`; the one-line shim
below adds no logic.
"""
from bct import rout_efficiency as _bct_rout_efficiency


def routing_efficiency(L):
    """Routing efficiency matrix: 1 / shortest-path length, 0 if disconnected.
    Identical call to communication_metrics.compute_all."""
    return _bct_rout_efficiency(L, transform=None)[1]


__all__ = ['routing_efficiency']
