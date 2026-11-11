#!/usr/bin/env python3
from bct import rout_efficiency as _bct_rout_efficiency


def routing_efficiency(L):
    """Routing efficiency matrix: 1 / shortest-path length, 0 if disconnected.
    Identical call to communication_metrics.compute_all."""
    return _bct_rout_efficiency(L, transform=None)[1]


__all__ = ['routing_efficiency']
