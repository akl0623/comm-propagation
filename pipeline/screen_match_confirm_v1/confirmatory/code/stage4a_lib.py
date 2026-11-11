#!/usr/bin/env python3
import os
import sys
import json
import time
import hashlib
import warnings

import numpy as np
import pandas as pd

SMCR = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/screen_match_confirm_v1'
sys.path.append(os.path.join(SMCR, 'confirmatory', 'code'))
sys.path.append(os.path.join(SMCR, 'candidate_screens', 'code'))

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

from bridge_noscreen import (  # Stage-0 engine (my reimplementation)
    P1Engine, cell_seed, METRICS, SEEDS, blind_from_paths, ob_from_paths,
    RAW, BI, UNI_P, FOLD_P,
)
import screens  # Stage-2 pure rule code (B5)
from screens import Path
from run_screens import gen_candidates  # Stage-2 candidate generation code

warnings.filterwarnings('ignore')

ALPHAS = np.logspace(-6, 1, 15)
L1_RATIOS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]
GRID120 = [{'alpha': float(a), 'l1_ratio': float(l)}
           for a in ALPHAS for l in L1_RATIOS]
assert len(GRID120) == 120
ELASTICNET_PARAMS = {'fit_intercept': True, 'max_iter': 200000, 'tol': 1e-8,
                     'selection': 'cyclic', 'random_state': 42}

ENGINE = None
GLOBAL = {}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def load_confirmatory_config():
    cfg = json.load(open(os.path.join(SMCR, 'config', 'CONFIRMATORY_SCREEN.json')))
    return cfg


_CONF = None


def conf():
    global _CONF
    if _CONF is None:
        _CONF = load_confirmatory_config()
    return _CONF


def build_engine():
    global ENGINE
    import bridge_noscreen as B
    eng = P1Engine()
    eng.load()
    ENGINE = eng
    B.ENGINE = eng   # the bridge feature builders read bridge_noscreen.ENGINE
    return eng


def feature_cell_seed(sid):
    """Frozen common.seed_derivation: SHA256(screen_params_version +
    screen_id + cell_id)[:8] as big-endian int."""
    ver = 'v1'
    sid_ = conf()['screen_id']
    return int.from_bytes(hashlib.sha256(
        f'{ver}{sid_}{sid}'.encode()).digest()[:8], 'big')


# ══════════════════════════════════════════════════════════════════
# B5 confirmatory screening (re-run from frozen rules)
# ══════════════════════════════════════════════════════════════════
def b5_params_from_config():
    cfg = load_confirmatory_config()
    p = dict(cfg['rules'])
    p['Ksel'] = cfg['ddvs']['Ksel']
    p['jaccard_threshold'] = cfg['ddvs']['jaccard_threshold']
    return p


def screen_one_universe(eng, matrices, uni, K0, b5p):
    """Returns (registry_rows, attrition_rows). Identical construction to the
    Stage-2 run_universe B5 path (for byte-identical reproduction)."""
    r2i = eng.r2i
    roi_order = eng.roi_order
    reg_rows = []
    att_rows = []
    for row_id, r in uni.iterrows():
        sid = r['sample_id']
        si = r2i[r['seed_roi']]
        ei = r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        paths = gen_candidates(eng, si, ei, L, 'B5', sid, K0)
        breakdown = {}
        accepted_pre = []
        for p in paths:
            reason = screens.b5_reject_reason(p, b5p, matrices)
            if reason is None:
                accepted_pre.append(p)
            else:
                breakdown[reason] = breakdown.get(reason, 0) + 1
        accepted_post = screens.screen_B5(paths, b5p, matrices)
        in_support = len(accepted_pre) >= (1 if L == 1 else 5)
        for p in accepted_post:
            nodes = p.nodes
            u = nodes[:-1]
            v = nodes[1:]
            reg_rows.append({
                'sample_id': sid,
                'path_id': int(row_id * 1_000_000 + p.rank),
                'path_rank': int(p.rank),
                'sample_row_id': int(row_id),
                'roi_sequence': '|'.join(roi_order[x] for x in nodes),
                'roi_id_sequence': '|'.join(str(int(x)) for x in nodes),
                'edge_set': ';'.join(f'{int(a)}-{int(b)}'
                                     for a, b in zip(u, v)),
                'L': L, 'accepted': True, 'reject_reason': '',
                'screen_id': 'B5'})
        att_rows.append({
            'sample_id': sid,
            'n_candidate': len(paths),
            'n_accepted': len(accepted_pre),
            'n_rejected': len(paths) - len(accepted_pre),
            'reject_reason_breakdown': json.dumps(breakdown, sort_keys=True),
            'in_support': bool(in_support), 'L': L,
            'seed_roi': r['seed_roi'], 'endpoint_roi': r['endpoint_roi']})
    return reg_rows, att_rows


# ══════════════════════════════════════════════════════════════════
# C4/C6/C8R feature construction (sample-level)
# ══════════════════════════════════════════════════════════════════
def build_sample_features(eng, uni_rows, K0):
    """Per-sample features for C4/C6/C8R using K0 paths sampled with the
    frozen feature seed. uni_rows: DataFrame with sample_id, seed_roi,
    endpoint_roi, path_length_edges (row order defines feature row order)."""
    blind_cols = [f'blind_{j:02d}' for j in range(56)]
    ob_cols = []
    for m in METRICS:
        for q in range(4):
            for agg in ('med', 'iqr'):
                ob_cols.append(f'ob_{m}_p{q}_{agg}')
    L_arr = uni_rows['path_length_edges'].values.astype(np.int64)
    ep = np.empty((len(uni_rows), 4), dtype=np.float64)
    blind = np.empty((len(uni_rows), 56), dtype=np.float64)
    ob = np.empty((len(uni_rows), 32), dtype=np.float64)
    n_paths = np.zeros(len(uni_rows), dtype=np.int64)
    for i, (_, r) in enumerate(uni_rows.iterrows()):
        si = eng.r2i[r['seed_roi']]
        ei = eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        cs = feature_cell_seed(r['sample_id'])
        ranks = eng.sample_ranks(si, ei, L, K0, cs)
        paths = eng.unrank_cell(si, ei, L, ranks)
        n_paths[i] = paths.shape[0]
        ep[i] = [eng.mats[m][si, ei] for m in METRICS]
        if paths.shape[0] == 0:
            blind[i] = np.nan
            ob[i] = np.nan
            continue
        blind[i] = blind_from_paths(paths, L)
        obd = ob_from_paths(paths, L)
        ob[i] = [obd[c] for c in ob_cols]
    # L one-hot over L_is_1..L_is_20; L > 20 (Task A only) gets an all-zero
    # row (no one-hot category defined beyond 20) — documented behavior
    onehot = np.zeros((len(uni_rows), 20), dtype=np.float64)
    ok = L_arr <= 20
    onehot[ok] = np.eye(20)[L_arr[ok] - 1]
    X_c4 = np.hstack([onehot, ep])
    X_c6 = np.hstack([onehot, ep, blind])
    X_c8r = np.hstack([onehot, ep, ob])
    return X_c4, X_c6, X_c8r, n_paths


# ══════════════════════════════════════════════════════════════════
# preprocessing + nested CV (Stage-4 spec: Imputer -> VarianceThreshold(0)
# -> StandardScaler, fit on training partition only)
# ══════════════════════════════════════════════════════════════════
class PreprocessingPipeline4A:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.vt = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()
        self.fitted_ = False

    def fit(self, X):
        X_imp = self.imputer.fit_transform(X)
        X_vt = self.vt.fit_transform(X_imp)
        self.scaler.fit(X_vt)
        self.fitted_ = True
        return self

    def transform(self, X):
        X_imp = self.imputer.transform(X)
        X_vt = self.vt.transform(X_imp)
        return self.scaler.transform(X_vt)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def select_best(cand_rows):
    best_score = np.inf
    best_row = None
    for row in cand_rows:
        m = row['mean_rmse']
        if not np.isfinite(m):
            continue
        if m < best_score - 1e-12:
            best_score = m
            best_row = row
        elif abs(m - best_score) < 1e-12:
            if row['alpha'] > best_row['alpha']:
                best_score = m
                best_row = row
            elif abs(row['alpha'] - best_row['alpha']) < 1e-15:
                if row['l1_ratio'] > best_row['l1_ratio']:
                    best_score = m
                    best_row = row
    return best_row, best_score


def inner_grid_search(X_train, y_train, groups_train):
    # degenerate-case guard (documented): fewer seed groups than 5 in an
    # outer-train split -> use the available group count (never triggered
    # in the real runs: outer-train always has >100 seed groups)
    n_grp = len(set(groups_train))
    n_splits = min(5, n_grp)
    icv = GroupKFold(n_splits=n_splits)
    splits = list(icv.split(X_train, y_train, groups_train))
    cands = {}
    for tr, va in splits:
        pp = PreprocessingPipeline4A()
        Xtr = pp.fit_transform(X_train[tr])
        Xva = pp.transform(X_train[va])
        ytr = y_train[tr]
        yva = y_train[va]
        for r in GRID120:
            key = f"{r['alpha']:.17g}|{r['l1_ratio']:.17g}"
            model = ElasticNet(alpha=r['alpha'], l1_ratio=r['l1_ratio'],
                               **ELASTICNET_PARAMS)
            model.fit(Xtr, ytr)
            yp = model.predict(Xva)
            rmse = float(np.sqrt(mean_squared_error(yva, yp)))
            cands.setdefault(key, {'alpha': r['alpha'], 'l1_ratio': r['l1_ratio'],
                                   'fold_rmses': []})['fold_rmses'].append(rmse)
    rows = []
    for key, c in cands.items():
        c['mean_rmse'] = float(np.mean(c['fold_rmses']))
        rows.append(c)
    best_row, best_score = select_best(rows)
    return {'alpha': best_row['alpha'], 'l1_ratio': best_row['l1_ratio']}, \
        best_score


def train_outer_fold(X_train, y_train, groups_train, X_test):
    if len(set(groups_train)) < 2:
        # degenerate fallback (documented): cannot run grouped inner CV with
        # a single seed group; use a fixed mid-grid HP. Never expected in the
        # real runs.
        best_params = {'alpha': 0.01, 'l1_ratio': 0.5}
        best_inner_rmse = np.nan
    else:
        best_params, best_inner_rmse = inner_grid_search(X_train, y_train, groups_train)
    pp = PreprocessingPipeline4A()
    X_tr = pp.fit_transform(X_train)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model = ElasticNet(alpha=best_params['alpha'],
                           l1_ratio=best_params['l1_ratio'],
                           **ELASTICNET_PARAMS)
        model.fit(X_tr, y_train)
        n_warn = sum(1 for x in w if issubclass(x.category, ConvergenceWarning))
    X_te = pp.transform(X_test)
    y_pred = model.predict(X_te)
    return {'y_pred': y_pred, 'best_alpha': best_params['alpha'],
            'best_l1_ratio': best_params['l1_ratio'],
            'best_inner_rmse': float(best_inner_rmse),
            'n_iter_': int(model.n_iter_),
            'n_warnings': int(n_warn)}


# ══════════════════════════════════════════════════════════════════
# metrics (frozen definitions from the bridge/statistics modules)
# ══════════════════════════════════════════════════════════════════
def metric_block(yt, yp):
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 3:
        return {k: np.nan for k in ('r2', 'rmse', 'mae', 'pearson_r',
                                    'calib_intercept', 'calib_slope')} | \
            {'n': int(mask.sum()), 'na_reason': 'n<3'}
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
