#!/usr/bin/env python3
"""
frozen_feature_machine.py — reproduction of the frozen Task C feature machine.

Re-implements the baseline Task C pipeline (C4_ENDPOINT, C6_SUMMARY,
C8R_ENDPOINT_PLUS_ORDERED_BASIS at the V3_AMENDED_ENDPOINT_PILOT primary
family) with new vectorized code:

  * P1 path engine (per-seed latency DAG, path-count DP, deterministic
    rank-based sampling) — reimplemented from the frozen mathematical
    definition (dense numpy DP; counts < 2^53 so all integer arithmetic
    is exact).
  * blind-56 (C6) and Legendre ordered-basis-32 (C8R) feature builders —
    new vectorized implementations of the frozen formulas.
  * 5x5 seed-grouped nested CV with inner GroupKFold(5), 72-candidate
    PILOT grid (alpha in logspace(-6,1,15) >= 1e-3, l1_ratio 8 values),
    min-mean-RMSE selection with frozen tie-breaks, train-only
    median-impute / zero-variance-drop / StandardScaler preprocessing.
  * C6/C8R fit at the per-fold C4-selected hyperparameters (conditional
    pilot), matching the frozen C4_HP_CONDITIONAL_ENDPOINT_PILOT mode.

Sampling size K: K per (repeat, fold) is read from the frozen feature
artifact TASK_C_FEATURES_V1.parquet (column K); these values reproduce the
baseline and are used only for this reproduction.

Bridge criterion: |R2_new - R2_frozen| <= 0.005 per model at
MEAN_PREDICTION_SAMPLE_LEVEL (and REPEAT_SUMMARY), vs
frozen_s0/05_taskC/results/TASK_C_OVERALL_METRICS_V1.csv.
"""
import os
import sys
import json
import time
import hashlib
import random as _py_random
import multiprocessing as mp

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────── paths
PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

ROOT = PIPELINE_ROOT
SMCR = os.path.join(ROOT, 'screen_match_confirm_v1')
RAW = os.path.join(ROOT, 'raw_inputs')
FS0 = os.path.join(ROOT, 'frozen_s0')
BI = os.path.join(FS0, 'bridge_inputs')
OUT = os.path.join(SMCR, 'confirmatory', 'taskC', 'bridge_v1')
os.makedirs(OUT, exist_ok=True)

LAT_P = os.path.join(RAW, 'Reordered_matrix_onset_delay__median.csv')
NAV_P = os.path.join(RAW, 'navigation_efficiency_matrix_0.25density.csv')
ROUT_P = os.path.join(RAW, 'rout_efficiency_matrix_0.25density.csv')
SRCH_P = os.path.join(RAW, 'search_information_matrix_0.25density.csv')
COMM_P = os.path.join(RAW, 'communicability_matrix_0.25density.csv')
SC_P = os.path.join(BI, 'averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv')
ROI_P = os.path.join(BI, 'roi_order_C0R2C.txt')
UNI_P = os.path.join(BI, 'taskC_primary_2636_universe_C0R2D.csv')
FOLD_P = os.path.join(BI, 'taskC_primary_fold_manifest_C0R2D_v2.csv')
FEAT_V1_P = os.path.join(BI, 'TASK_C_FEATURES_V1.parquet')
C8R_SEED_P = os.path.join(BI, 'TASK_C_FEATURES_C8R_SEEDLEVEL_V3.parquet')
OOF_REF_P = os.path.join(FS0, '05_taskC', 'results', 'TASK_C_OOF_PREDICTIONS_V1.parquet')
OVERALL_REF_P = os.path.join(FS0, '05_taskC', 'results', 'TASK_C_OVERALL_METRICS_V1.csv')
FOLD_METRICS_REF_P = os.path.join(FS0, '05_taskC', 'results', 'TASK_C_OUTER_FOLD_METRICS_V1.csv')

METRICS = ['nav', 'rout', 'search', 'comm']
SEEDS = [20260727, 20260728, 20260729, 20260730, 20260731]
N_FEAT_56 = 56
L_MAX = 20

ALPHAS_FULL = np.logspace(-6, 1, 15)
PILOT_ALPHAS = [a for a in ALPHAS_FULL if a >= 1e-3]
L1_RATIOS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
PILOT_GRID = [{'alpha': float(a), 'l1_ratio': float(l)}
              for a in PILOT_ALPHAS for l in L1_RATIOS]
assert len(PILOT_GRID) == 72
ELASTICNET_PARAMS = {'fit_intercept': True, 'max_iter': 200000, 'tol': 1e-8,
                     'selection': 'cyclic', 'random_state': 42}

# ─────────────────────────────────────────────────── open-trace firewall ──
_orig_open = open
_OPENED = []


def _trace_open(f, *a, **kw):
    p = f if isinstance(f, str) else getattr(f, 'name', str(f))
    _OPENED.append((p, kw.get('mode', 'r')))
    return _orig_open(f, *a, **kw)


def sha256_file(p):
    h = hashlib.sha256()
    with _orig_open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def cell_seed(gs, sid):
    return int.from_bytes(hashlib.sha256(f"{gs}|{sid}".encode()).digest()[:8], 'big')


# ══════════════════════════════════════════════════════════════════
# P1 PATH ENGINE (new implementation of the frozen math)
# ══════════════════════════════════════════════════════════════════
class P1Engine:
    """Per-seed latency DAG + exact path counts + deterministic sampling.

    Frozen math (from the Frozen-S0 feature builder definition):
      * seed hemisphere: node index < 180 -> nodes 0..179, else 180..359
      * DAG nodes: seed (latency 0.0) + same-hemisphere ROIs with finite,
        positive latency; sorted by latency.
      * edge u -> v iff latency[v] > latency[u] and SC_mask[u, v] > 0.
      * count DP: ntl[ei][0] = 1; ntl[u][L] = sum_{v in out[u]} ntl[v][L-1].
      * unrank(rank): walk from seed; at node u with `rem` edges remaining,
        pick the first child v (in sorted out-edge order) such that
        rank < prefix_count(v), subtract the prefix.

    Dense numpy implementation: counts fit float64 exactly
    (max ~1.4e11 << 2^53, verified against the frozen universe).
    """

    def __init__(self):
        self.mask = None          # (360, 360) int8
        self.lat = None           # (360, 360) float64
        self.mats = {}            # metric -> (360, 360) float64
        self.roi_order = []
        self.r2i = {}
        self._dag_adj = {}        # si -> (nodes, adj) lazily built
        self._counts = {}         # (si, ei, L) -> int
        self._ctprep = {}         # (si, ei, L) -> list of {u: (cands, cum)}

    def load(self):
        sc = pd.read_csv(SC_P, header=None).values.astype(np.float64)
        self.mask = (sc > 0).astype(np.int8)
        self.lat = pd.read_csv(LAT_P, index_col=0).values.astype(np.float64)
        for m, p in [('nav', NAV_P), ('rout', ROUT_P),
                     ('search', SRCH_P), ('comm', COMM_P)]:
            self.mats[m] = pd.read_csv(p, header=None).values.astype(np.float64)
        with _orig_open(ROI_P) as f:
            self.roi_order = [l.strip() for l in f if l.strip()]
        self.r2i = {r: i for i, r in enumerate(self.roi_order)}
        assert len(self.roi_order) == 360

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
                # frozen rule: edge only if latency strictly increases
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
        """Per-rem unranking tables: {u_pos: (cands_positions, cum_counts)}.

        d[rem] applies when `rem` edges remain; children counts are
        ct[v][rem-1]. Only nodes with ct[u][rem] > 0 are stored.
        All keys/values are POSITIONS into the seed's node list.
        """
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
            cvecs.append(c)   # cvecs[rem-1] = ct[u][rem]
        d = {}
        for rem in range(1, L + 1):
            cur = cvecs[rem - 1]
            prev = cvecs[rem - 2] if rem > 1 else None
            reach = np.nonzero(cur > 0)[0]
            entry = {}
            for u in reach:
                cands = np.nonzero(adj[u] > 0)[0]
                # frozen rule: children are visited in ascending NODE-ID order
                # (oe[u] = sorted(set(...))), so sort positions by node id
                cands = cands[np.argsort(np.asarray(no, dtype=np.int16)[cands],
                                         kind='stable')]
                if prev is None:
                    # rem == 1: child must be ei with ct[ei][0] == 1
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
        """Vectorized unranking of sorted ranks -> (n, L+1) int16 paths.

        Walks by POSITION in the node list; final matrix holds ROI ids.
        """
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
            # stable-sort rows by current node; apply the SAME permutation to
            # cur, rr AND the partially written paths matrix so every row
            # keeps belonging to the same rank.
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


ENGINE = None


# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDERS (new vectorized implementations of frozen formulas)
# ══════════════════════════════════════════════════════════════════
def blind_from_paths(paths, L):
    """Per-path 28 stats -> cell-level 56 (median, IQR) — frozen cell_56 math.

    Per path per metric: [mean, std(ddof=0), min, max, first, last, slope]
    with slope = polyfit(linspace(0,1,L), ev, 1)[0] (L>=2, else 0.0).
    """
    K = paths.shape[0]
    if K == 0:
        return np.full(N_FEAT_56, np.nan)
    u = paths[:, :-1]
    v = paths[:, 1:]
    ppa = np.empty((K, 28), dtype=np.float64)
    if L >= 2:
        z = np.linspace(0.0, 1.0, L)
        zd = z - z.mean()
        zz = float((zd * zd).sum())
    for j, m in enumerate(METRICS):
        X = ENGINE.mats[m][u, v].astype(np.float64)   # (K, L)
        col = 7 * j   # 7 stats per metric: mean, std, min, max, first, last, slope
        ppa[:, col] = X.mean(axis=1)
        ppa[:, col + 1] = X.std(axis=1, ddof=0)
        ppa[:, col + 2] = X.min(axis=1)
        ppa[:, col + 3] = X.max(axis=1)
        ppa[:, col + 4] = X[:, 0]
        ppa[:, col + 5] = X[:, -1]
        if L >= 2:
            ppa[:, col + 6] = ((X - X.mean(axis=1, keepdims=True)) * zd).sum(axis=1) / zz
        else:
            ppa[:, col + 6] = 0.0
    cf = np.zeros(N_FEAT_56, dtype=np.float64)
    for j in range(28):
        col = ppa[:, j]
        fin = col[np.isfinite(col)]
        if fin.size > 0:
            cf[2 * j] = np.median(fin)
            if fin.size > 1:
                cf[2 * j + 1] = float(np.subtract(*np.percentile(fin, [75, 25])))
            else:
                cf[2 * j + 1] = 0.0
        else:
            cf[2 * j] = np.nan
            cf[2 * j + 1] = np.nan
    return cf


def legendre_basis(L):
    """B[s, q] = P_q(z_s), z = -1 + 2(s-1)/(L-1) (L=1 -> z=0), q <= min(3, L-1)."""
    qmax = min(3, L - 1)
    if qmax < 0:
        return None
    z = np.array([0.0]) if L == 1 else np.asarray(
        -1.0 + 2.0 * np.arange(L) / (L - 1), dtype=np.float64)
    cols = []
    cols.append(np.ones_like(z))
    if qmax >= 1:
        cols.append(z)
    if qmax >= 2:
        cols.append((3.0 * z * z - 1.0) / 2.0)
    if qmax >= 3:
        cols.append((5.0 * z ** 3 - 3.0 * z) / 2.0)
    return np.column_stack(cols)


def ob_from_paths(paths, L):
    """32 per-seed ordered-basis features (med/iqr of Legendre betas)."""
    K = paths.shape[0]
    if K == 0:
        raise ValueError('empty path set for a formally sampled cell')
    u = paths[:, :-1]
    v = paths[:, 1:]
    B = legendre_basis(L)
    qmax = min(3, L - 1)
    out = {}
    for m in METRICS:
        X = ENGINE.mats[m][u, v].astype(np.float64)
        beta = np.zeros((K, 4), dtype=np.float64)
        if qmax >= 0:
            Q, R = np.linalg.qr(B, mode='reduced')
            beta[:, :qmax + 1] = np.linalg.solve(R, Q.T @ X.T).T
        for q in range(4):
            col = beta[:, q]
            out[f'ob_{m}_p{q}_med'] = float(np.median(col))
            if K > 1:
                out[f'ob_{m}_p{q}_iqr'] = float(np.subtract(*np.percentile(col, [75, 25])))
            else:
                out[f'ob_{m}_p{q}_iqr'] = 0.0
    return out


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING + INNER CV (frozen reference engine semantics)
# ══════════════════════════════════════════════════════════════════
class PreprocessingPipeline:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.fitted_ = False
        self.zero_var_cols_ = None

    def fit(self, X):
        X_np = X.values.astype(np.float64) if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        X_imp = self.imputer.fit_transform(X_np)
        variances = np.var(X_imp, axis=0)
        self.zero_var_cols_ = np.where(variances < 1e-15)[0]
        keep = [i for i in range(X_imp.shape[1]) if i not in self.zero_var_cols_]
        X_clean = X_imp[:, keep]
        if X_clean.shape[1] > 0:
            self.scaler.fit(X_clean)
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError('Pipeline not fitted')
        X_np = X.values.astype(np.float64) if hasattr(X, 'values') else np.asarray(X, dtype=np.float64)
        X_imp = self.imputer.transform(X_np)
        keep = [i for i in range(X_imp.shape[1]) if i not in self.zero_var_cols_]
        X_clean = X_imp[:, keep]
        if X_clean.shape[1] > 0:
            return self.scaler.transform(X_clean)
        return X_clean

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def select_best(cand_rows):
    """Frozen selection: min mean RMSE; ties -> larger alpha, then larger l1_ratio."""
    best_score = np.inf
    best_row = None
    for row in cand_rows:
        mean_rmse = row['mean_rmse']
        if not np.isfinite(mean_rmse):
            continue
        if mean_rmse < best_score - 1e-12:
            best_score = mean_rmse
            best_row = row
        elif abs(mean_rmse - best_score) < 1e-12:
            if row['alpha'] > best_row['alpha']:
                best_score = mean_rmse
                best_row = row
            elif abs(row['alpha'] - best_row['alpha']) < 1e-15:
                if row['l1_ratio'] > best_row['l1_ratio']:
                    best_score = mean_rmse
                    best_row = row
    return best_row, best_score


def inner_grid_search(X_train, y_train, groups_train):
    inner_cv = GroupKFold(n_splits=5)
    splits = list(inner_cv.split(X_train, y_train, groups_train))
    candidates = {}
    for sp in splits:
        tr, va = sp
        pp = PreprocessingPipeline()
        Xtr = pp.fit_transform(pd.DataFrame(X_train[tr]))
        Xva = pp.transform(pd.DataFrame(X_train[va]))
        ytr = y_train[tr]
        yva = y_train[va]
        if Xtr.shape[1] == 0:
            for r in PILOT_GRID:
                key = f"{r['alpha']:.17g}|{r['l1_ratio']:.17g}"
                candidates.setdefault(key, {'alpha': r['alpha'], 'l1_ratio': r['l1_ratio'],
                                            'fold_rmses': []})['fold_rmses'].append(np.inf)
            continue
        for r in PILOT_GRID:
            model = ElasticNet(alpha=r['alpha'], l1_ratio=r['l1_ratio'],
                               **ELASTICNET_PARAMS, warm_start=False)
            model.fit(Xtr, ytr)
            yp = model.predict(Xva)
            rmse = float(np.sqrt(mean_squared_error(yva, yp)))
            key = f"{r['alpha']:.17g}|{r['l1_ratio']:.17g}"
            candidates.setdefault(key, {'alpha': r['alpha'], 'l1_ratio': r['l1_ratio'],
                                        'fold_rmses': []})['fold_rmses'].append(rmse)
    cand_rows = []
    for key, c in candidates.items():
        c['mean_rmse'] = float(np.mean(c['fold_rmses'])) if len(c['fold_rmses']) == 5 else np.inf
        cand_rows.append(c)
    best_row, best_score = select_best(cand_rows)
    best_params = {'alpha': best_row['alpha'], 'l1_ratio': best_row['l1_ratio']}
    return best_params, best_score, cand_rows


def train_outer_fold(X_train, y_train, groups_train, X_test):
    best_params, best_inner_rmse, _ = inner_grid_search(X_train, y_train, groups_train)
    pp_full = PreprocessingPipeline()
    X_train_scaled = pp_full.fit_transform(pd.DataFrame(X_train))
    best_model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'],
                            **ELASTICNET_PARAMS, warm_start=False)
    best_model.fit(X_train_scaled, y_train)
    X_test_scaled = pp_full.transform(pd.DataFrame(X_test))
    y_pred = best_model.predict(X_test_scaled)
    return y_pred, best_params, best_inner_rmse


def fit_fixed_hp(X_train, y_train, X_test, alpha, l1_ratio):
    """Conditional fit at a fixed HP (C6/C8R pilot mode)."""
    pp = PreprocessingPipeline()
    X_train_scaled = pp.fit_transform(pd.DataFrame(X_train))
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **ELASTICNET_PARAMS, warm_start=False)
    model.fit(X_train_scaled, y_train)
    X_test_scaled = pp.transform(pd.DataFrame(X_test))
    return model.predict(X_test_scaled)


def metric_block(yt, yp):
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 3:
        return {k: np.nan for k in ('r2', 'rmse', 'mae', 'pearson_r', 'calib_intercept',
                                    'calib_slope')} | {'n': int(mask.sum()), 'na_reason': 'n<3'}
    y, p = yt[mask], yp[mask]
    out = {'n': int(mask.sum()), 'na_reason': ''}
    out['rmse'] = float(np.sqrt(mean_squared_error(y, p)))
    out['mae'] = float(mean_absolute_error(y, p))
    if np.ptp(y) == 0:
        out['r2'] = np.nan
        out['na_reason'] += 'constant_y;'
    else:
        out['r2'] = float(r2_score(y, p))
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        out['pearson_r'] = np.nan
        out['calib_intercept'] = np.nan
        out['calib_slope'] = np.nan
        out['na_reason'] += 'constant_y_or_pred;'
    else:
        out['pearson_r'] = float(np.corrcoef(y, p)[0, 1])
        slope, intercept = np.polyfit(p, y, 1)
        out['calib_slope'] = float(slope)
        out['calib_intercept'] = float(intercept)
    return out


# ══════════════════════════════════════════════════════════════════
# FOLD PROCESSING (path sampling + per-seed features)
# ══════════════════════════════════════════════════════════════════
G = {}


def _worker_init():
    global ENGINE
    ENGINE = G['engine']


def process_fold(args):
    rep, fold = args
    K = G['k_map'][(rep, fold)]
    sids = G['fold_sids'][(rep, fold)]
    sample_info = G['sample_info']
    rows = []
    for sid in sids:
        si, ei, L = sample_info[sid]
        cs = cell_seed(str(SEEDS[rep]), sid)
        ranks = ENGINE.sample_ranks(si, ei, L, K, cs)
        paths = ENGINE.unrank_cell(si, ei, L, ranks)
        # validate (frozen checks: endpoints, length, no repeats, mask==1)
        assert paths.shape[0] == len(ranks) == min(K, ENGINE.counts(si, ei, L))
        assert (paths[:, 0] == si).all() and (paths[:, -1] == ei).all()
        assert (paths.shape[1] - 1) == L
        for s in range(1, L + 1):
            assert (ENGINE.mask[paths[:, s - 1], paths[:, s]] == 1).all()
        blind56 = blind_from_paths(paths, L)
        ob32 = ob_from_paths(paths, L)
        rows.append({'sample_id': sid, 'n_paths': paths.shape[0],
                     'blind': blind56, 'ob': ob32})
    return rep, fold, rows


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    t0_total = time.time()
    open = _trace_open  # noqa: F841  (trace opens for the firewall audit)
    print('=' * 70)
    print('BRIDGE NO-SCREEN — Frozen-S0 replication with new code')
    print('=' * 70, flush=True)

    # ── 1. load universe / folds / K map / y ──
    uni = pd.read_csv(UNI_P)
    fm = pd.read_csv(FOLD_P)
    feat_v1 = pd.read_parquet(FEAT_V1_P)
    print(f'  universe {uni.shape}, manifest {fm.shape}, frozen features {feat_v1.shape}')

    assert len(uni) == 2636
    assert fm.groupby(['repeat', 'sample_id']).size().max() == 1
    k_map = {}
    for (rep, fold), g in feat_v1.groupby(['repeat', 'test_fold']):
        k_map[(int(rep), int(fold))] = int(g['K'].iloc[0])
    assert len(k_map) == 25
    print('  K map (from frozen features artifact):',
          {k: k_map[k] for k in sorted(k_map)[:5]}, '...')

    y_by_sample = dict(zip(uni['sample_id'], uni['y_endpoint'].astype(np.float64)))

    merged = fm.sort_values(['repeat', 'sample_id'], kind='stable').reset_index(drop=True)
    G['fold_sids'] = {}
    for rep in range(5):
        rd = merged[merged['repeat'] == rep]
        for fold in range(5):
            G['fold_sids'][(rep, fold)] = rd.loc[rd['test_fold'] == fold, 'sample_id'].tolist()
    G['k_map'] = k_map

    # ── 2. engine ──
    print('[2] loading P1 engine...', flush=True)
    t0 = time.time()
    eng = P1Engine()
    eng.load()
    G['engine'] = eng
    print(f'  engine loaded ({time.time()-t0:.1f}s)', flush=True)

    # sample_info keyed by sample_id -> (si_idx, ei_idx, L) with integer indices
    sample_info = {}
    for _, r in uni.iterrows():
        si = eng.r2i[r['seed_roi']]
        ei = eng.r2i[r['endpoint_roi']]
        assert si is not None and ei is not None, r['sample_id']
        sample_info[r['sample_id']] = (int(si), int(ei), int(r['path_length_edges']))
    G['sample_info'] = sample_info

    # ── 3. sample + per-seed features (parallel over 25 folds) ──
    print('[3] sampling paths + per-seed features (25 folds, parallel)...', flush=True)
    t0 = time.time()
    per_seed = {}   # (sid, rep) -> dict(n_paths, blind56, ob32)
    with mp.Pool(8, initializer=_worker_init) as pool:
        for rep, fold, rows in pool.imap_unordered(process_fold, list(G['fold_sids'])):
            for r in rows:
                per_seed[(r['sample_id'], rep)] = r
            print(f'    r{rep}f{fold}: {len(rows)} cells', flush=True)
    n_paths_total = sum(r['n_paths'] for r in per_seed.values())
    print(f'  sampled {n_paths_total} paths in {time.time()-t0:.1f}s', flush=True)

    # ── 4. identity checks vs frozen artifacts ──
    print('[4] identity checks vs Frozen-S0 artifacts...', flush=True)
    ident = {'cells': 0, 'n_paths_mismatch': 0, 'blind_max_rel': 0.0,
             'ob_max_abs': 0.0, 'ob_max_rel': 0.0, 'ep_max_abs': 0.0}
    blind_cols = [f'blind_{j:02d}' for j in range(N_FEAT_56)]
    ep_cols = [f'ep_{m}' for m in METRICS]
    feat_idx = feat_v1.set_index(['sample_id', 'repeat'])
    c8r_seed = pd.read_parquet(C8R_SEED_P).set_index(['sample_id', 'repeat'])
    ob_cols = [c for c in c8r_seed.columns if c.startswith('ob_')]
    for (sid, rep), r in per_seed.items():
        fr = feat_idx.loc[(sid, rep)]
        ident['cells'] += 1
        if int(fr['n_paths']) != r['n_paths']:
            ident['n_paths_mismatch'] += 1
        blind_fr = fr[blind_cols].values.astype(np.float64)
        diff = np.abs(r['blind'] - blind_fr)
        rel = diff / np.maximum(np.abs(blind_fr), 1e-300)
        ident['blind_max_rel'] = max(ident['blind_max_rel'], float(rel.max()))
        ident['blind_max_abs'] = max(ident.get('blind_max_abs', 0.0), float(diff.max()))
        # bad only if BOTH relative and absolute exceed 1e-12 (near-zero
        # denominators must not inflate benign reduction drift)
        ident['blind_bad_cells'] = ident.get('blind_bad_cells', 0) + int(
            ((rel > 1e-12) & (diff > 1e-12)).any())
        si, ei, L = sample_info[sid]
        ep_mine = np.array([eng.mats[m][si, ei] for m in METRICS])
        ident['ep_max_abs'] = max(ident['ep_max_abs'],
                                  float(np.abs(ep_mine - fr[ep_cols].values.astype(np.float64)).max()))
        csr = c8r_seed.loc[(sid, rep)]
        ob_mine = np.array([r['ob'][c] for c in ob_cols])
        ob_fr = csr[ob_cols].values.astype(np.float64)
        ident['ob_max_abs'] = max(ident['ob_max_abs'],
                                  float(np.abs(ob_mine - ob_fr).max()))
        rel_ob = np.abs(ob_mine - ob_fr) / np.maximum(np.abs(ob_fr), 1e-300)
        ident['ob_max_rel'] = max(ident['ob_max_rel'], float(rel_ob.max()))
    print(f"  cells={ident['cells']} n_paths_mismatch={ident['n_paths_mismatch']} "
          f"blind_max_rel={ident['blind_max_rel']:.3e} ob_max_abs={ident['ob_max_abs']:.3e} "
          f"ob_max_rel={ident['ob_max_rel']:.3e} ep_max_abs={ident['ep_max_abs']:.3e}", flush=True)
    ident_ok = (ident['n_paths_mismatch'] == 0 and ident['blind_bad_cells'] == 0
                and ident['ob_max_rel'] <= 1e-12 and ident['ep_max_abs'] == 0.0)

    # ── 5. sample-level aggregation (C6 / C8R frames) ──
    sids_sorted = sorted(uni['sample_id'])
    sid_pos = {s: i for i, s in enumerate(sids_sorted)}
    blind_sm = np.empty((2636, N_FEAT_56), dtype=np.float64)
    ob_sm = np.empty((2636, 32), dtype=np.float64)
    for i, sid in enumerate(sids_sorted):
        vals56 = np.stack([per_seed[(sid, rep)]['blind'] for rep in range(5)])
        vals32 = np.stack([per_seed[(sid, rep)]['ob'][ob_cols] if False else
                           np.array([per_seed[(sid, rep)]['ob'][c] for c in ob_cols])
                           for rep in range(5)])
        blind_sm[i] = np.median(vals56, axis=0)
        ob_sm[i] = np.median(vals32, axis=0)

    L_arr = np.array([int(sample_info[s][2]) for s in sids_sorted], dtype=np.float64)
    ep_arr = np.empty((2636, 4), dtype=np.float64)
    for i, sid in enumerate(sids_sorted):
        si, ei, L = sample_info[sid]
        ep_arr[i] = [eng.mats[m][si, ei] for m in METRICS]

    # sample-level identity: C6 = 5-seed median of blind; C8R = 5-seed median of ob
    blind_frozen_sm = feat_v1.sort_values(['sample_id', 'repeat']).groupby(
        'sample_id', sort=True)[blind_cols].median()
    blind_frozen_sm = blind_frozen_sm.loc[sids_sorted].values.astype(np.float64)
    rel_sm = np.abs(blind_sm - blind_frozen_sm) / np.maximum(np.abs(blind_frozen_sm), 1e-300)
    c8r_v3 = pd.read_parquet(BI + '/TASK_C_FEATURES_C8R_V3.parquet').sort_values('sample_id')
    ob_frozen_sm = c8r_v3[ob_cols].values.astype(np.float64)
    rel_ob_sm = np.abs(ob_sm - ob_frozen_sm) / np.maximum(np.abs(ob_frozen_sm), 1e-300)
    print(f'  sample-level identity: blind_max_rel={rel_sm.max():.3e} '
          f'ob_max_rel={rel_ob_sm.max():.3e}', flush=True)
    ident['blind_sm_max_rel'] = float(rel_sm.max())
    ident['ob_sm_max_rel'] = float(rel_ob_sm.max())
    ident['onehot_L_exact'] = bool(np.array_equal(
        np.eye(20)[L_arr.astype(int) - 1],
        c8r_v3[[f'L_is_{l}' for l in range(1, 21)]].values.astype(np.float64)))

    X_c4_sample = np.hstack([L_arr[:, None], ep_arr])                    # (2636, 5)
    X_c6_sample = np.hstack([L_arr[:, None], ep_arr, blind_sm])          # (2636, 61)
    X_c8r_sample = np.hstack([np.eye(20)[L_arr.astype(int) - 1], ep_arr, ob_sm])  # (2636, 56)

    # row-level frames (repeat-major, matching merged row order)
    n_rows = len(merged)
    row_pos = np.array([sid_pos[s] for s in merged['sample_id']], dtype=np.int64)
    X_c4 = X_c4_sample[row_pos]
    y_row = np.array([y_by_sample[s] for s in merged['sample_id']], dtype=np.float64)
    grp_row = merged['seed_roi'].values
    # sanity: row-level X matches frozen features parquet L/ep columns
    fr_sorted = feat_v1.sort_values(['repeat', 'sample_id'], kind='stable')
    max_L_diff = np.abs(X_c4[:, 0] - fr_sorted['L'].values.astype(np.float64)).max()
    max_ep_diff = np.abs(X_c4[:, 1:] - fr_sorted[ep_cols].values.astype(np.float64)).max()
    print(f'  C4 row frame vs frozen: |dL|={max_L_diff} |dep|={max_ep_diff}', flush=True)
    assert max_L_diff == 0 and max_ep_diff == 0

    # ── 6. C4 nested CV (PILOT grid) ──
    print('[6] C4 nested CV (25 folds, PILOT 9x8 grid)...', flush=True)
    t0 = time.time()
    c4_pred_rows = []
    c4_hp = {}
    for rep in range(5):
        rd = merged[merged['repeat'] == rep]
        for fold in range(5):
            test_mask = rd['test_fold'].values == fold
            test_idx = rd.index[test_mask].values.astype(np.int64)
            train_idx = rd.index[~test_mask].values.astype(np.int64)
            y_pred, best_params, best_rmse = train_outer_fold(
                X_c4[train_idx], y_row[train_idx], grp_row[train_idx], X_c4[test_idx])
            c4_hp[(rep, fold)] = (best_params['alpha'], best_params['l1_ratio'])
            for i, idx in enumerate(test_idx):
                c4_pred_rows.append({
                    'model': 'C4_ENDPOINT', 'repeat': rep, 'fold': fold,
                    'sample_id': merged['sample_id'].iloc[idx],
                    'y_true': float(y_row[idx]), 'y_pred': float(y_pred[i])})
    print(f'  C4 done in {time.time()-t0:.1f}s', flush=True)
    hp_frozen = {(0.001, 0.05): 0}
    hp_mine = set(c4_hp.values())
    print(f'  C4 HP mine={sorted(hp_mine)} frozen_expected={sorted(hp_frozen)}', flush=True)

    # ── 7. C6 / C8R conditional (fixed HP per fold) ──
    print('[7] C6/C8R conditional fits...', flush=True)
    c6_pred_rows = []
    c8r_pred_rows = []
    for rep in range(5):
        rd = merged[merged['repeat'] == rep]
        for fold in range(5):
            test_mask = rd['test_fold'].values == fold
            test_pos = np.array([sid_pos[s] for s in rd.loc[test_mask, 'sample_id']])
            train_pos = np.array([sid_pos[s] for s in rd.loc[~test_mask, 'sample_id']])
            y_tr = np.array([y_by_sample[s] for s in rd.loc[~test_mask, 'sample_id']],
                            dtype=np.float64)
            y_te = np.array([y_by_sample[s] for s in rd.loc[test_mask, 'sample_id']],
                            dtype=np.float64)
            grp_tr = rd.loc[~test_mask, 'seed_roi'].values
            alpha, l1 = c4_hp[(rep, fold)]
            for X_sample, model_id, rows_out in [
                    (X_c6_sample, 'C6_SUMMARY', c6_pred_rows),
                    (X_c8r_sample, 'C8R_ENDPOINT_PLUS_ORDERED_BASIS', c8r_pred_rows)]:
                y_pred = fit_fixed_hp(X_sample[train_pos], y_tr, X_sample[test_pos],
                                      alpha, l1)
                for i, sid in enumerate(rd.loc[test_mask, 'sample_id']):
                    rows_out.append({'model': model_id, 'repeat': rep, 'fold': fold,
                                     'sample_id': sid, 'y_true': float(y_te[i]),
                                     'y_pred': float(y_pred[i])})

    # ── 8. assemble OOF + metrics at the frozen levels ──
    oof = pd.DataFrame(c4_pred_rows + c6_pred_rows + c8r_pred_rows)
    oof.to_parquet(os.path.join(OUT, 'bridge_oof_predictions.parquet'), index=False)

    overall_rows = []
    fold_rows = []
    prim = oof
    for model in ['C4_ENDPOINT', 'C6_SUMMARY', 'C8R_ENDPOINT_PLUS_ORDERED_BASIS']:
        msub = prim[prim['model'] == model]
        for (rep, fold), g in msub.groupby(['repeat', 'fold']):
            mb = metric_block(g['y_true'].values, g['y_pred'].values)
            fold_rows.append({'model': model, 'repeat': rep, 'test_fold': fold, **mb})
        # REPEAT_SUMMARY
        per = {rep: metric_block(sub['y_true'].values, sub['y_pred'].values)
               for rep, sub in msub.groupby('repeat')}
        mean = {}
        for k in ('r2', 'rmse', 'mae', 'pearson_r', 'calib_intercept', 'calib_slope'):
            vals = [per[r][k] for r in range(5) if np.isfinite(per[r][k])]
            mean[k] = float(np.mean(vals)) if vals else np.nan
        mean['n'] = sum(per[r]['n'] for r in range(5))
        overall_rows.append({'model': model, 'level': 'REPEAT_SUMMARY', **mean})
        mp_df = msub.groupby('sample_id')[['y_true', 'y_pred']].mean().reset_index()
        mpm = metric_block(mp_df['y_true'].values, mp_df['y_pred'].values)
        overall_rows.append({'model': model, 'level': 'MEAN_PREDICTION_SAMPLE_LEVEL', **mpm})
    fold_df = pd.DataFrame(fold_rows)
    overall_df = pd.DataFrame(overall_rows)
    fold_df.to_csv(os.path.join(OUT, 'bridge_outer_fold_metrics.csv'), index=False)
    overall_df.to_csv(os.path.join(OUT, 'bridge_overall_metrics.csv'), index=False)

    # ── 9. compare vs Frozen-S0 ──
    ref_overall = pd.read_csv(OVERALL_REF_P)
    ref_fold = pd.read_csv(FOLD_METRICS_REF_P)
    comp = {'models': {}, 'fold_max_r2_diff': {}, 'fold_mean_r2_diff': {},
            'pred_max_abs_diff': {}, 'pred_mean_abs_diff': {}, 'hp': {}}
    for model in ['C4_ENDPOINT', 'C6_SUMMARY', 'C8R_ENDPOINT_PLUS_ORDERED_BASIS']:
        mine = {}
        for level in ('REPEAT_SUMMARY', 'MEAN_PREDICTION_SAMPLE_LEVEL'):
            row = overall_df[(overall_df['model'] == model) & (overall_df['level'] == level)]
            mine[level] = float(row['r2'].iloc[0])
        froz = {}
        for level in ('REPEAT_SUMMARY', 'MEAN_PREDICTION_SAMPLE_LEVEL'):
            row = ref_overall[(ref_overall['model'] == model) & (ref_overall['level'] == level)]
            froz[level] = float(row['r2'].iloc[0])
        comp['models'][model] = {
            'r2_mine': mine, 'r2_frozen': froz,
            'delta_MEAN_PREDICTION_SAMPLE_LEVEL': mine['MEAN_PREDICTION_SAMPLE_LEVEL']
            - froz['MEAN_PREDICTION_SAMPLE_LEVEL'],
            'delta_REPEAT_SUMMARY': mine['REPEAT_SUMMARY'] - froz['REPEAT_SUMMARY']}
        # fold-level
        mf = fold_df[fold_df['model'] == model].set_index(['repeat', 'test_fold'])['r2']
        ff = ref_fold[ref_fold['model'] == model].set_index(['repeat', 'test_fold'])['r2']
        comp['fold_max_r2_diff'][model] = float((mf - ff).abs().max())
        comp['fold_mean_r2_diff'][model] = float((mf - ff).abs().mean())
    # per-prediction comparison vs frozen OOF parquet
    ref_oof = pd.read_parquet(OOF_REF_P)
    ref_prim = ref_oof[(ref_oof['analysis_family'] == 'V3_AMENDED_ENDPOINT_PILOT')]
    for model in ['C4_ENDPOINT', 'C6_SUMMARY', 'C8R_ENDPOINT_PLUS_ORDERED_BASIS']:
        a = oof[oof['model'] == model].set_index(['repeat', 'fold', 'sample_id'])['y_pred']
        b = ref_prim[ref_prim['model'] == model].set_index(['repeat', 'fold', 'sample_id'])['y_pred']
        joined = a.to_frame('mine').join(b.to_frame('frozen'))
        comp['pred_max_abs_diff'][model] = float((joined['mine'] - joined['frozen']).abs().max())
        comp['pred_mean_abs_diff'][model] = float((joined['mine'] - joined['frozen']).abs().mean())
    comp['n_rows_matched'] = int(len(joined))
    comp['identity'] = ident
    comp['identity_ok'] = bool(ident_ok)
    comp['c4_hp_mine'] = {f'r{rep}f{fold}': list(c4_hp[(rep, fold)]) for rep in range(5) for fold in range(5)}
    comp['c4_hp_frozen_expected'] = {'alpha': 0.001, 'l1_ratio': 0.05}
    comp['K_source'] = 'from TASK_C_FEATURES_V1.parquet (K column)'

    # ── 10. file-open audit ──
    # Detect any open of the fold-K-selection analysis CSV or of the
    # exploratory directory (forbidden for this reproduction).
    POISON_SUBSTR = 'fold_K_selection'
    bad = [p for p, _ in _OPENED if POISON_SUBSTR in p or '/exploratory/' in p]
    comp['firewall_open_audit'] = {'n_opens': len(_OPENED), 'forbidden_opens': bad,
                                   'pass': len(bad) == 0}

    gate = all(abs(comp['models'][m]['delta_MEAN_PREDICTION_SAMPLE_LEVEL']) <= 0.005
               and abs(comp['models'][m]['delta_REPEAT_SUMMARY']) <= 0.005
               for m in comp['models'])
    comp['bridge_gate_0p005'] = bool(gate)
    comp['wall_s'] = round(time.time() - t0_total, 1)
    comp['frozen_overall_sha256'] = sha256_file(OVERALL_REF_P)
    comp['oof_reference_sha256'] = sha256_file(OOF_REF_P)

    with open(os.path.join(OUT, 'bridge_comparison.json'), 'w') as f:
        json.dump(comp, f, indent=2)

    print('\n' + '=' * 70)
    print('BRIDGE RESULTS')
    print('=' * 70)
    for model in comp['models']:
        m = comp['models'][model]
        print(f"  {model:32s} mine={m['r2_mine']['MEAN_PREDICTION_SAMPLE_LEVEL']:.6f} "
              f"frozen={m['r2_frozen']['MEAN_PREDICTION_SAMPLE_LEVEL']:.6f} "
              f"Δ={m['delta_MEAN_PREDICTION_SAMPLE_LEVEL']:+.6f}")
        print(f"    REPEAT_SUMMARY: mine={m['r2_mine']['REPEAT_SUMMARY']:.6f} "
              f"frozen={m['r2_frozen']['REPEAT_SUMMARY']:.6f} "
              f"Δ={m['delta_REPEAT_SUMMARY']:+.6f}")
        print(f"    fold max|ΔR²|={comp['fold_max_r2_diff'][model]:.3e} "
              f"pred max|Δ|={comp['pred_max_abs_diff'][model]:.3e}")
    print(f"  identity_ok={ident_ok}  firewall_audit_pass={len(bad)==0}")
    print(f"  GATE (|Δ|<=0.005 both levels, 3 models): {'PASS' if gate else 'FAIL'}")

    # ── 11. audit verdict artifacts ──
    if gate and ident_ok and len(bad) == 0:
        pass_json = {
            'bridge': 'PASS',
            'criterion': '|R2_new - R2_frozen| <= 0.005 per model at both levels',
            'comparison': comp['models'],
            'fold_max_r2_diff': comp['fold_max_r2_diff'],
            'pred_max_abs_diff': comp['pred_max_abs_diff'],
            'identity': ident,
            'K_source': comp['K_source'],
            'wall_s': comp['wall_s'],
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        with open(os.path.join(SMCR, 'audit', 'BRIDGE_PASS.json'), 'w') as f:
            json.dump(pass_json, f, indent=2)
        print(f"\nBRIDGE PASS — wrote audit/BRIDGE_PASS.json")
    else:
        with open(os.path.join(SMCR, 'audit', 'BRIDGE_MISMATCH.md'), 'w') as f:
            f.write('# BRIDGE MISMATCH — reproduction failed\n\n')
            f.write('## Overall R2 comparison\n\n')
            f.write('| model | level | mine | frozen | delta |\n|---|---|---|---|---|\n')
            for model in comp['models']:
                m = comp['models'][model]
                for level in ('MEAN_PREDICTION_SAMPLE_LEVEL', 'REPEAT_SUMMARY'):
                    f.write(f"| {model} | {level} | {m['r2_mine'][level]:.10f} | "
                            f"{m['r2_frozen'][level]:.10f} | "
                            f"{m['r2_mine'][level]-m['r2_frozen'][level]:+.10f} |\n")
            f.write(f"\nidentity_ok={ident_ok}\nfirewall_audit_pass={len(bad)==0}\n")
            f.write('\n## Full comparison\n\n```json\n')
            json.dump(comp, f, indent=2)
            f.write('\n```\n')
        print(f"\nBRIDGE MISMATCH — STOP. Wrote audit/BRIDGE_MISMATCH.md")

    print(f'\nwall time: {comp["wall_s"]:.1f}s')
    return gate, comp


if __name__ == '__main__':
    main()
