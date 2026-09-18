#!/usr/bin/env python3
"""
screens.py — B1..B6 outcome-blind biological screens (Stage 2).

FIREWALL: this module performs NO file I/O.  Every function is a pure
function of (paths, params, matrices): structural/temporal/anatomical
inputs only.  No target, conditional_prob, OOF, prediction, label or
MRI data may ever reach these functions.

Screen signatures (frozen):
    screen_Bk(paths: List[Path], params: dict, matrices: Matrices) -> List[Path]
returns the ACCEPTED paths after the screen rules + DDVS diversity
downsampling.  The companion b*_reject_reason() functions give the
per-path reject reason (None = accept) for the attrition tables.

Shared DDVS (frozen): exact-duplicate removal by ROI sequence; edge-Jaccard
complete-link clustering at jaccard_threshold; one medoid per cluster
(max mean intra-cluster Jaccard, tie -> lexicographic ROI sequence);
farthest-first selection down to Ksel (first = lexicographically smallest;
distance = 1 - edge Jaccard; ties broken by whitelist edge bonus when
provided, then lexicographic).

Lobe mapping is public HCP MMP1.0 atlas metadata (Glasser et al. 2016);
identical content to the Stage-1 copy (duplicated here so that Stage 2
never reads the exploratory tree).
"""
from collections import namedtuple
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

Path = namedtuple('Path', ['nodes', 'sample_id', 'rank'])
# nodes: tuple of int ROI ids (0..359); sample_id: str; rank: int >= 0

METRICS = ['nav', 'rout', 'search', 'comm']

# ──────────────────────────────────────────────────────────────────
# LOBE MAPPING — public HCP MMP1.0 atlas metadata (Glasser et al. 2016)
# ──────────────────────────────────────────────────────────────────
LOBES = {}
_OCC = ('V1 V2 V3 V3A V3B V3CD V4 V4t V6 V6A V7 V8 VMV1 VMV2 VMV3 LO1 LO2 LO3 '
        'MT MST FST POS1 POS2').split()
_TMP = ('FFC VVC PIT PH PHT PHA1 PHA2 PHA3 TE1a TE1m TE1p TE2a TE2p TF TGd TGv '
        'PeEc EC PreS ProS H A1 A4 A5 PBelt MBelt LBelt RI STGa STSva STSvp '
        'STSda STSdp TPOJ1 TPOJ2 TPOJ3 PSL SFL PCV STV TA2 DVT').split()
_PAR = ('1 2 3a 3b 5L 5m 5mv 7AL 7Am 7m 7PC 7PL 7Pm IPS1 LIPd LIPv VIP MIP '
        'IP0 IP1 IP2 PGp PGs PGi PF PFm PFop PFt PI AIP').split()
_CIN = ('23c 23d 24dd 24dv 25 31a 31pd 31pv 33pr a24 a24pr a32pr d23ab d32 '
        'p24 p24pr p32 p32pr s32 v23ab RSC').split()
_FRO = ('4 6a 6d 6ma 6mp 6r 6v 55b SCEF FEF PEF 8Ad 8Av 8BL 8BM 8C 9-46d 9a '
        '9m 9p 10d 10pp 10r 10v 11l 13l a10p a47r 44 45 46 47l 47m 47s a9-46v '
        'i6-8 IFJa IFJp IFSp IFSa p9-46v p10p p47r s6-8 OFC pOFC').split()
_INS = ('MI PI Ig Pir AVI AAIC FOP1 FOP2 FOP3 FOP4 FOP5 OP1 OP2-3 OP4 43 52 '
        'RI PoI1 PoI2 PFcm PFop').split()
for _a in _OCC:
    LOBES[_a] = 'Occipital'
for _a in _TMP:
    LOBES[_a] = 'Temporal'
for _a in _PAR:
    LOBES[_a] = 'Parietal'
for _a in _CIN:
    LOBES[_a] = 'Cingulate'
for _a in _FRO:
    LOBES[_a] = 'Frontal'
for _a in _INS:
    LOBES[_a] = 'Insula'
LOBE_ORDER = ['Frontal', 'Parietal', 'Occipital', 'Temporal', 'Cingulate', 'Insula']


class Matrices:
    """Structural inputs handed to every screen (built by the driver).

    All arrays are 360x360 float64 unless stated otherwise:
      sc_mask (int8), sc_weight, tract_length, onset (row=seed),
      nav, rout, search, comm,
      yeo7_ids (360,), lobe_ids (360,), lobe_names (360,),
      relay_rois (set of roi ids from the anatomical whitelist, role=relay),
      whitelist_edge_set (set of (u,v) int pairs whose endpoints share a
      whitelist circuit)
    """


# ──────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────
def path_edge_set(p):
    return frozenset((int(p.nodes[s - 1]), int(p.nodes[s]))
                     for s in range(1, len(p.nodes)))


def _to_arrays(paths):
    """(n, L+1) int32 node matrix + L. Assumes uniform L within a cell."""
    n = len(paths)
    L = len(paths[0].nodes) - 1
    arr = np.empty((n, L + 1), dtype=np.int32)
    for i, p in enumerate(paths):
        arr[i] = p.nodes
    return arr, L


def _node_onset(nodes, matrices):
    """Onset of each node on a path, with DAG semantics: the seed node has
    onset 0.0 by construction (the raw matrix diagonal lat[si,si] is NOT the
    seed's onset and must not be used)."""
    si = int(nodes[0])
    vals = matrices.onset[si][nodes].astype(np.float64).copy()
    vals[0] = 0.0
    return vals


def _jac_matrix(paths):
    """Pairwise edge-Jaccard matrix for a list of paths (vectorized)."""
    n = len(paths)
    edges = sorted({e for p in paths for e in path_edge_set(p)})
    if not edges:
        return np.ones((n, n), dtype=np.float64)
    e2i = {e: i for i, e in enumerate(edges)}
    E = np.zeros((n, len(edges)), dtype=np.int8)
    for i, p in enumerate(paths):
        for e in path_edge_set(p):
            E[i, e2i[e]] = 1
    inter = E.astype(np.float64) @ E.T.astype(np.float64)
    deg = E.sum(axis=1, dtype=np.float64)
    J = inter / np.maximum(deg[:, None] + deg[None, :] - inter, 1e-12)
    np.fill_diagonal(J, 1.0)
    return J


def _farthest_first(paths, K, J=None, bonus=None):
    """Farthest-first (distance = 1 - edge Jaccard); first = lexicographically
    smallest ROI sequence; ties broken by whitelist bonus then lexicographic."""
    if len(paths) <= K:
        return list(paths)
    if J is None:
        J = _jac_matrix(paths)
    n = len(paths)
    order = sorted(range(n), key=lambda k: paths[k].nodes)
    sel = [order[0]]
    sel_set = {order[0]}
    while len(sel) < K:
        best = None
        best_d = -1.0
        best_bonus = -1
        for k in order:
            if k in sel_set:
                continue
            dmin = min(1.0 - J[k, s] for s in sel)
            b = bonus[k] if bonus is not None else 0
            if (best is None or dmin > best_d + 1e-12
                    or (abs(dmin - best_d) <= 1e-12
                        and (b > best_bonus or (b == best_bonus
                                                and paths[k].nodes < paths[best].nodes)))):
                best = k
                best_d = dmin
                best_bonus = b
        sel.append(best)
        sel_set.add(best)
    return [paths[k] for k in sel]


def ddvs_downsample(paths, Ksel, jaccard_threshold, bonus=None):
    """Frozen DDVS: dedup -> complete-link edge-Jaccard clusters at
    jaccard_threshold -> medoid per cluster -> farthest-first to Ksel."""
    if not paths:
        return []
    # 1. exact-duplicate removal (by ROI sequence; keep first occurrence)
    seen = {}
    for p in paths:
        if p.nodes not in seen:
            seen[p.nodes] = p
    uniq = list(seen.values())
    if len(uniq) == 1:
        return uniq
    # 2. pairwise edge Jaccard
    J = _jac_matrix(uniq)
    # 3. complete-link clustering at threshold
    n = len(uniq)
    Z = linkage(squareform(1.0 - J, checks=False), method='complete')
    labels = fcluster(Z, t=1.0 - jaccard_threshold, criterion='distance')
    # 4. medoid per cluster (max mean intra-cluster Jaccard; tie lexicographic)
    medoids = []
    for cl in np.unique(labels):
        idx = np.where(labels == cl)[0]
        if len(idx) == 1:
            medoids.append(uniq[int(idx[0])])
            continue
        sub = J[np.ix_(idx, idx)]
        mean_j = sub.mean(axis=1)
        cand = [uniq[int(k)] for k in idx[mean_j == mean_j.max()]]
        medoids.append(min(cand, key=lambda p: p.nodes))
    medoids.sort(key=lambda p: p.nodes)
    # 5. farthest-first down to Ksel
    if len(medoids) <= Ksel:
        return medoids
    return _farthest_first(medoids, Ksel, bonus=bonus)


# ══════════════════════════════════════════════════════════════════
# B1 — ANATOMICAL NETWORK CONSISTENCY
# ══════════════════════════════════════════════════════════════════
def b1_reject_reason(path, params, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    L = len(nodes) - 1
    # 1. all edges in the seed hemisphere
    hem0 = nodes[0] < 180
    if not ((nodes < 180) == hem0).all():
        return 'CROSS_HEMISPHERE'
    # 2. Yeo7 transitions <= yeo7_transition_max (L=1 -> 0)
    y = matrices.yeo7_ids[nodes]
    if int((y[1:] != y[:-1]).sum()) > params['yeo7_transition_max']:
        return 'TOO_MANY_YEO7_TRANSITIONS'
    # 3. lobe oscillation (immediate A->B->A) unless the middle ROI is a
    #    known relay from the anatomical whitelist
    if params.get('lobe_oscillation_check', True) and L >= 2:
        lb = matrices.lobe_ids[nodes]
        for i in range(L - 1):
            if lb[i] == lb[i + 2] and lb[i] != lb[i + 1]:
                if int(nodes[i + 1]) not in matrices.relay_rois:
                    return 'LOBE_OSCILLATION'
    return None


def screen_B1(paths, params, matrices):
    accepted = [p for p in paths if b1_reject_reason(p, params, matrices) is None]
    bonus = None
    if params.get('whitelist_bonus_in_downsampling', True):
        bonus = [sum(1 for s in range(1, len(p.nodes))
                     if (p.nodes[s - 1], p.nodes[s]) in matrices.whitelist_edge_set)
                 for p in accepted]
    return ddvs_downsample(accepted, params['Ksel'], params['jaccard_threshold'],
                           bonus=bonus)


# ══════════════════════════════════════════════════════════════════
# B2 — CONDUCTION VELOCITY PLAUSIBILITY
# ══════════════════════════════════════════════════════════════════
def b2_reject_reason(path, params, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    vals = _node_onset(nodes, matrices)
    u = nodes[:-1]
    v = nodes[1:]
    d = matrices.tract_length[u, v]
    dl = vals[1:] - vals[:-1]
    bad_lat = np.where(dl <= 0)[0]
    if len(bad_lat):
        return 'NONPOSITIVE_LATENCY_DIFF'
    vel = d / dl
    lo = params['velocity_lower_m_per_s']
    hi = params['velocity_upper_m_per_s']
    if ((vel < lo) | (vel > hi)).any():
        return 'VELOCITY_OUT_OF_RANGE'
    return None


def screen_B2(paths, params, matrices):
    accepted = [p for p in paths if b2_reject_reason(p, params, matrices) is None]
    return ddvs_downsample(accepted, params['Ksel'], params['jaccard_threshold'])


# ══════════════════════════════════════════════════════════════════
# B3 — TEMPORAL ROBUSTNESS (TC-DDVS)
# ══════════════════════════════════════════════════════════════════
def b3_reject_reason(path, params, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    vals = _node_onset(nodes, matrices)
    # RAW_STRICT: strictly increasing, no binning
    if (np.diff(vals) <= 0).any():
        return 'RAW_NONSTRICT'
    n_ok = 1
    for w in params['bin_widths_ms']:
        bins = np.floor(vals / w).astype(np.int64)
        # legal in the binned representation iff no serial edge shares a bin
        if (np.diff(bins) > 0).all():
            n_ok += 1
    if n_ok >= 1 + params['min_legal_bins']:
        return None
    return 'INSUFFICIENT_BIN_CONSISTENCY'


def screen_B3(paths, params, matrices):
    accepted = [p for p in paths if b3_reject_reason(p, params, matrices) is None]
    return ddvs_downsample(accepted, params['Ksel'], params['jaccard_threshold'])


# ══════════════════════════════════════════════════════════════════
# B4 — SC STRENGTH GRADIENT
# ══════════════════════════════════════════════════════════════════
def b4_reject_reason(path, params, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    u = nodes[:-1]
    v = nodes[1:]
    mean_sc = float(matrices.sc_weight[u, v].mean())
    if mean_sc < params['threshold_mean_sc']:
        return 'SC_BELOW_40TH_PERCENTILE'
    return None


def screen_B4(paths, params, matrices):
    accepted = [p for p in paths if b4_reject_reason(p, params, matrices) is None]
    return ddvs_downsample(accepted, params['Ksel'], params['jaccard_threshold'])


# ══════════════════════════════════════════════════════════════════
# B5 — PATH DIVERSITY DDVS (PS-1 baseline)
# ══════════════════════════════════════════════════════════════════
def b5_reject_reason(path, params, matrices):
    """Hard validity (frozen PS-1 step 1)."""
    nodes = np.asarray(path.nodes, dtype=np.int64)
    L = len(nodes) - 1
    if len(set(nodes.tolist())) != L + 1:
        return 'REPEATED_ROI'
    vals = _node_onset(nodes, matrices)
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


def screen_B5(paths, params, matrices):
    accepted = [p for p in paths if b5_reject_reason(p, params, matrices) is None]
    return ddvs_downsample(accepted, params['Ksel'], params['jaccard_threshold'])


# ══════════════════════════════════════════════════════════════════
# B6 — COMMUNICATION METRIC COHERENCE
# ══════════════════════════════════════════════════════════════════
def b6_reject_reason(path, params, matrices):
    nodes = np.asarray(path.nodes, dtype=np.int64)
    u = nodes[:-1]
    v = nodes[1:]
    for m in METRICS:
        x = getattr(matrices, m)[u, v]
        if (not np.isfinite(x).all()) or (x < 0).any():
            return 'METRIC_NONFINITE'
    nav = matrices.nav[u, v]
    rout = matrices.rout[u, v]
    ratio = float(nav.mean() / rout.mean())
    if ratio < params['nav_rout_ratio_lower'] or ratio > params['nav_rout_ratio_upper']:
        return 'NAV_ROUT_RATIO_EXTREME'
    return None


def screen_B6(paths, params, matrices):
    accepted = [p for p in paths if b6_reject_reason(p, params, matrices) is None]
    return ddvs_downsample(accepted, params['Ksel'], params['jaccard_threshold'])
