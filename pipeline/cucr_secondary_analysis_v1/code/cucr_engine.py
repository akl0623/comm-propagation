#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd
import random as _py_random

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
RAW = os.path.join(ROOT, 'raw_inputs')
FROZEN_ROI = os.path.join(ROOT, 'frozen_s0', 'bridge_inputs', 'roi_order_C0R2C.txt')

SC_P = os.path.join(RAW, 'averageConnectivity_tractLength_0.25density.csv')
LAT_P = os.path.join(os.environ.get('CUCR_MATRIX_DIR', os.path.dirname(ROOT)), 'median_onset_time_matrix_251114.txt')
NAV_P = os.path.join(RAW, 'navigation_efficiency_matrix_0.25density.csv')
ROUT_P = os.path.join(RAW, 'rout_efficiency_matrix_0.25density.csv')
SRCH_P = os.path.join(RAW, 'search_information_matrix_0.25density.csv')
COMM_P = os.path.join(RAW, 'communicability_matrix_0.25density.csv')

METRICS = ['nav', 'rout', 'search', 'comm']
K0_SAMPLE = 512


class CucrEngine:
    """CUCR-input twin of the frozen P1Engine (same public API and math)."""

    def __init__(self):
        self.mask = None          # (360, 360) int8
        self.lat = None           # (360, 360) float64
        self.mats = {}            # metric -> (360, 360) float64
        self.roi_order = []
        self.r2i = {}
        self._dag_adj = {}
        self._counts = {}
        self._ctprep = {}

    def load(self):
        sc = pd.read_csv(SC_P, header=None).values.astype(np.float64)
        # PRE-DECLARED symmetrization: mask-union (mirror of C0R2C).
        self.mask = ((sc > 0) | (sc > 0).T).astype(np.int8)
        self.lat = np.loadtxt(LAT_P, dtype=np.float64)
        for m, p in [('nav', NAV_P), ('rout', ROUT_P),
                     ('search', SRCH_P), ('comm', COMM_P)]:
            self.mats[m] = pd.read_csv(p, header=None).values.astype(np.float64)
        with open(FROZEN_ROI) as f:
            self.roi_order = [l.strip() for l in f if l.strip()]
        self.r2i = {r: i for i, r in enumerate(self.roi_order)}
        assert len(self.roi_order) == 360
        assert self.mask.shape == (360, 360)
        assert self.lat.shape == (360, 360)

    def _dag(self, si):
        if si in self._dag_adj:
            return self._dag_adj[si]
        hemi = list(range(0, 180)) if si < 180 else list(range(180, 360))
        nodes = [(si, 0.0)]
        for ri in hemi:
            if ri == si:
                continue
            tv = self.lat[si, ri]
            if np.isnan(tv) or tv <= 0:
                continue
            nodes.append((ri, float(tv)))
        nodes.sort(key=lambda x: x[1])
        no = [n[0] for n in nodes]
        lat_pos = np.array([n[1] for n in nodes], dtype=np.float64)
        n = len(no)
        adj = np.zeros((n, n), dtype=np.float64)
        for i, ui in enumerate(no):
            for j in range(i + 1, n):
                vj = no[j]
                if lat_pos[j] > lat_pos[i] and self.mask[ui, vj]:
                    adj[i, j] = 1.0
        self._dag_adj[si] = (no, adj)
        return no, adj

    def counts(self, si, ei, L):
        """Total number of seed->endpoint paths of exactly L edges (exact int)."""
        key = (si, ei, L)
        if key in self._counts:
            return self._counts[key]
        no, adj = self._dag(si)
        if ei not in set(no):
            self._counts[key] = 0
            return 0
        n = len(no)
        ei_pos = no.index(ei)
        c = np.zeros(n, dtype=np.float64)
        c[ei_pos] = 1.0
        for _ in range(L):
            c = adj @ c
        T = int(c[no.index(si)])
        assert T == float(c[no.index(si)]), "count lost exactness"
        self._counts[key] = T
        return T

    def ct_prep(self, si, ei, L):
        key = (si, ei, L)
        if key in self._ctprep:
            return self._ctprep[key]
        no, adj = self._dag(si)
        n = len(no)
        ei_pos = no.index(ei)
        cvecs = []
        c = np.zeros(n, dtype=np.float64)
        c[ei_pos] = 1.0
        for rem in range(1, L + 1):
            c = adj @ c
            cvecs.append(c)
        d = {}
        for rem in range(1, L + 1):
            cur = cvecs[rem - 1]
            prev = cvecs[rem - 2] if rem > 1 else None
            reach = np.nonzero(cur > 0)[0]
            entry = {}
            for u in reach:
                cands = np.nonzero(adj[u] > 0)[0]
                cands = cands[np.argsort(np.asarray(no, dtype=np.int16)[cands],
                                         kind='stable')]
                if prev is None:
                    vals = (cands == ei_pos).astype(np.float64)
                else:
                    vals = prev[cands]
                keep = vals > 0
                cvals = vals[keep]
                cc = cands[keep]
                entry[int(u)] = (cc.astype(np.int32),
                                 np.cumsum(cvals.astype(np.float64)))
            d[rem] = entry
        self._ctprep[key] = d
        return d

    def unrank_cell(self, si, ei, L, ranks):
        n = len(ranks)
        if n == 0:
            return np.empty((0, L + 1), dtype=np.int16)
        d = self.ct_prep(si, ei, L)
        no, _ = self._dag(si)
        si_pos = no.index(si)
        cur = np.full(n, si_pos, dtype=np.int32)
        rr = np.asarray(ranks, dtype=np.int64).copy()
        paths = np.empty((n, L + 1), dtype=np.int16)
        paths[:, 0] = si
        rem = L
        while rem > 0:
            entry = d[rem]
            order = np.argsort(cur, kind='stable')
            cur_s = cur[order]
            rr_s = rr[order]
            paths = paths[order]
            newcur = np.empty(n, dtype=np.int32)
            newrr = rr_s.copy()
            k = 0
            while k < n:
                u = int(cur_s[k])
                end = k + int(np.searchsorted(cur_s[k:], u, side='right'))
                cands, cum = entry[u]
                pos = np.searchsorted(cum, rr_s[k:end], side='right')
                newcur[k:end] = cands[pos]
                prev = np.where(pos > 0, cum[pos - 1], 0.0)
                newrr[k:end] = rr_s[k:end] - prev.astype(np.int64)
                k = end
            paths[:, L - rem + 1] = np.asarray(no, dtype=np.int16)[newcur]
            cur = newcur
            rr = newrr
            rem -= 1
        return paths

    def sample_ranks(self, si, ei, L, K, cs):
        T = self.counts(si, ei, L)
        if T == 0:
            return []
        n = min(K, T)
        rng = _py_random.Random(cs)
        if T <= 2 ** 62:
            return sorted(rng.sample(range(T), n))
        s = set()
        for i in range(T - n, T):
            t = rng.randint(0, i + 1)
            s.add(t if t not in s else i)
        return sorted(s)


# ───────────────────────────────────────────── candidate generation (verbatim)
def cell_seed(screen_id, sid):
    """run_screens.cell_seed (candidate sampling seed; NOT the C8R feature
    seed).  SHA256('{screen_id}|{sid}')[:8], no 'v1' prefix."""
    import hashlib
    return int.from_bytes(
        hashlib.sha256(f'{screen_id}|{sid}'.encode()).digest()[:8], 'big')


def feature_seed(screen_id, sid):
    """so_lib.feature_seed (C8R feature-path seed, used in Stage 2 only)."""
    import hashlib
    return int.from_bytes(
        hashlib.sha256(f'v1{screen_id}{sid}'.encode()).digest()[:8], 'big')


def random_walk_candidates(eng, si, ei, L, n_target, screen_id, sid):
    """Seeded random walk for L > 10 cells (frozen fallback included)."""
    seed_int = cell_seed(screen_id, sid)
    d = eng.ct_prep(si, ei, L)
    no, _ = eng._dag(si)
    si_pos = no.index(si)
    rng = np.random.default_rng(seed_int)
    seen = set()
    out = []
    max_attempts = max(20000, 200 * n_target)
    attempts = 0
    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        cur = si_pos
        path = [si]
        ok = True
        for rem in range(L, 0, -1):
            entry = d[rem]
            if cur not in entry:
                ok = False
                break
            cands, _cum = entry[cur]
            k = int(rng.integers(0, len(cands)))
            cur = int(cands[k])
            path.append(int(no[cur]))
        if not ok:
            continue
        t = tuple(path)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    if len(out) < n_target:
        T = eng.counts(si, ei, L)
        rng2 = _py_random.Random(seed_int)
        ranks = sorted(rng2.sample(range(T), n_target - len(out)))
        for r in ranks:
            p = tuple(int(x) for x in eng.unrank_cell(si, ei, L, [r])[0])
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def gen_candidates(eng, si, ei, L, screen_id, sid, K0):
    """Verbatim gen_candidates from candidate_screens/code/run_screens.py."""
    T = eng.counts(si, ei, L)
    if T == 0:
        return []
    if L <= 10:
        if T <= K0:
            ranks = list(range(T))          # full enumeration
        else:
            ranks = eng.sample_ranks(si, ei, L, K0,
                                     cell_seed(screen_id, sid))
        arr = eng.unrank_cell(si, ei, L, ranks)
        return [tuple(int(x) for x in arr[i]) for i in range(len(ranks))]
    return random_walk_candidates(eng, si, ei, L, min(K0, T), screen_id, sid)


# ─────────────────────────────────────────── matrix container (minimal twin)
class CucrMatrices:
    """sc_mask + onset + four metric attributes (same attribute names as the
    frozen make_matrices output used by cell_path_stats)."""

    def __init__(self, eng):
        self.sc_mask = eng.mask
        self.onset = eng.lat
        for m in METRICS:
            setattr(self, m, eng.mats[m])


def cell_path_stats(eng, matrices, si, ei, L, screen_id, sid, K0=K0_SAMPLE):
    """(nodes, b5_valid, variance_stat) — verbatim so_lib.cell_path_stats."""
    paths = gen_candidates(eng, si, ei, L, screen_id, sid, K0)
    if not paths:
        return (np.empty((0, L + 1), dtype=np.int64), np.zeros(0, bool),
                np.zeros(0))
    nodes = np.asarray(paths, dtype=np.int64)
    K = nodes.shape[0]
    u, v = nodes[:, :-1], nodes[:, 1:]
    srt = np.sort(nodes, axis=1)
    ok = ~(np.diff(srt, axis=1) == 0).any(axis=1)          # no repeated ROI
    ok &= (matrices.sc_mask[u, v] == 1).all(axis=1)        # SC edge present
    vals = matrices.onset[si][nodes].astype(np.float64).copy()
    vals[:, 0] = 0.0
    ok &= (np.diff(vals, axis=1) > 0).all(axis=1)          # strictly temporal
    for m in METRICS:
        ok &= np.isfinite(getattr(matrices, m)[u, v]).all(axis=1)
    if L < 2:
        return nodes, ok, np.full(K, np.nan)
    stat = np.zeros(K, dtype=np.float64)
    for m in METRICS:
        stat += np.var(getattr(matrices, m)[u, v], axis=1, ddof=0) / 4.0
    return nodes, ok, stat
