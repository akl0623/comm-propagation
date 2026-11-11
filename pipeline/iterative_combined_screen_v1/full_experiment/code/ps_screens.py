#!/usr/bin/env python3
import numpy as np

METRICS = ['nav', 'rout', 'search', 'comm']
S10_THRESHOLD = 0.41549280879846406   # frozen constant (Task C universe)
VEL_LO = 1.0
VEL_HI = 150.0


def b5_hard_validity(path, matrices):
    """Frozen B5 hard validity (screens.b5_reject_reason equivalent)."""
    nodes = np.asarray(path.nodes, dtype=np.int64)
    L = len(nodes) - 1
    if len(set(nodes.tolist())) != L + 1:
        return 'REPEATED_ROI'
    si = int(nodes[0])
    vals = matrices.onset[si][nodes].astype(np.float64).copy()
    vals[0] = 0.0
    for s in range(1, L + 1):
        u, v = int(nodes[s - 1]), int(nodes[s])
        if matrices.sc_mask[u, v] != 1:
            return 'SC_EDGE_ABSENT'
        if vals[s] <= vals[s - 1]:
            return 'NONSTRICT_TEMPORAL'
        for m in METRICS:
            x = getattr(matrices, m)[u, v]
            if not np.isfinite(x):
                return 'METRIC_NONFINITE'
    return None


def s10_component(path, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    L = len(nodes) - 1
    if L < 2:
        return None                      # degenerate variance, exempt
    u, v = nodes[:-1], nodes[1:]
    stat = 0.0
    for m in METRICS:
        x = getattr(matrices, m)[u, v]
        stat += float(np.var(x, ddof=0)) / 4.0
    if stat < S10_THRESHOLD:
        return 'VARIANCE_BELOW_MEDIAN'
    return None


def s14_component(path, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    u, v = nodes[:-1], nodes[1:]
    si = int(nodes[0])
    vals = matrices.onset[si][nodes].astype(np.float64).copy()
    vals[0] = 0.0
    dlat = vals[1:] - vals[:-1]          # > 0 guaranteed by B5 validity
    tl = matrices.tract_length[u, v]
    ok = tl > 0
    if ok.any():
        med = float(np.median(tl[ok] / dlat[ok]))
        if med < VEL_LO or med > VEL_HI:
            return 'MEDIAN_VELOCITY_OUT_OF_RANGE'
    return None


def screen_reject(screen_id, path, matrices):
    r = b5_hard_validity(path, matrices)
    if r is not None:
        return r
    if screen_id == 'S10':
        return s10_component(path, matrices)
    if screen_id == 'S14':
        return s14_component(path, matrices)
    raise ValueError(screen_id)


def screen_accept(screen_id, paths, matrices):
    """Accepted paths (reject-level, pre-DDVS) for a candidate list."""
    return [p for p in paths if screen_reject(screen_id, p, matrices) is None]
