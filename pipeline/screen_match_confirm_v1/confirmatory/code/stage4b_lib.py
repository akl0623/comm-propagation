#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import warnings

import numpy as np
import pandas as pd

ROOT = os.environ.get('PIPELINE_ROOT', '/mnt/d/revision_v2')
SMCR = os.path.join(ROOT, 'screen_match_confirm_v1')
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.join(PACKAGE_ROOT, 'pipeline')
sys.path[:0] = [
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1',
                 'confirmatory', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1',
                 'candidate_screens', 'code'),
]

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

import stage4a_lib as A4
from stage4a_lib import (PreprocessingPipeline4A, metric_block,
                         ELASTICNET_PARAMS, GRID120, inner_grid_search,
                         feature_cell_seed, build_engine)

warnings.filterwarnings('ignore')

TC = os.path.join(SMCR, 'confirmatory', 'taskC')
TA = os.path.join(SMCR, 'confirmatory', 'taskA')
INF = os.path.join(SMCR, 'confirmatory', 'inference')
STAB = os.path.join(SMCR, 'confirmatory', 'stability')
NC = os.path.join(SMCR, 'confirmatory', 'negative_controls')
VAL = os.path.join(SMCR, 'confirmatory', 'validation')
for d in (INF, STAB, NC, VAL):
    os.makedirs(d, exist_ok=True)

MODELS = ['C4', 'C6', 'C8R']
BOOT_SEED = 8769577528861415417   # frozen Task C cluster-bootstrap convention


def load_taskC_stage4a():
    """Returns (oof, fold_metrics, hp, support_meta, fm_sup, X, y, sid_pos)."""
    oof = pd.read_parquet(os.path.join(TC, 'OOF_PREDICTIONS.parquet'))
    fold_metrics = pd.read_csv(os.path.join(TC, 'OUTER_FOLD_METRICS.csv'))
    hp = pd.read_csv(os.path.join(TC, 'INNER_HYPERPARAMETERS.csv'))
    support = pd.read_csv(os.path.join(SMCR, 'confirmatory', 'support',
                                       'COMMON_SUPPORT.csv'))
    uni = pd.read_csv(A4.UNI_P)
    fm = pd.read_csv(A4.FOLD_P)
    support_sids = set(support[support['in_support']]['sample_id'])
    uni_sup = uni[uni['sample_id'].isin(support_sids)].reset_index(drop=True)
    uni_sup = uni_sup[['sample_id', 'seed_roi', 'endpoint_roi',
                       'path_length_edges']]
    eng = build_engine()
    K0 = A4.conf()['common_params']['K0_sample']
    X_c4, X_c6, X_c8r, _ = A4.build_sample_features(eng, uni_sup, K0)
    y = uni_sup['sample_id'].map(
        dict(zip(uni['sample_id'], uni['y_endpoint'].astype(np.float64)))
    ).values.astype(np.float64)
    sid_pos = {sid: i for i, sid in enumerate(uni_sup['sample_id'])}
    fm_sup = fm[fm['sample_id'].isin(support_sids)]
    return {'oof': oof, 'fold_metrics': fold_metrics, 'hp': hp,
            'uni_sup': uni_sup, 'eng': eng, 'X': {'C4': X_c4, 'C6': X_c6,
                                                  'C8R': X_c8r},
            'y': y, 'sid_pos': sid_pos, 'fm_sup': fm_sup,
            'fm': fm, 'uni': uni, 'K0': K0}


def fit_fixed_hp(X_train, y_train, X_test, alpha, l1_ratio):
    """Conditional refit at a fixed HP (Stage-4A-selected)."""
    pp = PreprocessingPipeline4A()
    X_tr = pp.fit_transform(X_train)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **ELASTICNET_PARAMS)
        model.fit(X_tr, y_train)
        n_warn = sum(1 for x in w if issubclass(x.category, ConvergenceWarning))
    return model.predict(pp.transform(X_test)), int(model.n_iter_), int(n_warn)


def run_cv_fixed_hp(ctx, y_values, model, hp):
    """Refit one model at the Stage-4A per-fold HPs on target y_values;
    returns per-(repeat,fold) predictions dict."""
    Xm = ctx['X'][model]
    fm_sup = ctx['fm_sup']
    sid_pos = ctx['sid_pos']
    hp_m = hp[hp['model'] == model]
    out = {}
    for rep in range(5):
        rd = fm_sup[fm_sup['repeat'] == rep]
        for fold in range(5):
            test_sids = rd.loc[rd['test_fold'] == fold, 'sample_id'].tolist()
            train_sids = rd.loc[rd['test_fold'] != fold, 'sample_id'].tolist()
            tr = np.array([sid_pos[s] for s in train_sids], dtype=np.int64)
            te = np.array([sid_pos[s] for s in test_sids], dtype=np.int64)
            row = hp_m[(hp_m['repeat'] == rep) & (hp_m['outer_fold'] == fold)]
            alpha = float(row['best_alpha'].iloc[0])
            l1 = float(row['best_l1_ratio'].iloc[0])
            yp, _, _ = fit_fixed_hp(Xm[tr], y_values[tr], Xm[te], alpha, l1)
            for i, sid in enumerate(test_sids):
                out[(sid, rep, fold)] = float(yp[i])
    return out


def repeat_summary_r2(oof_rows):
    """Mean over the 5 per-repeat pooled R2 for a single model's OOF rows."""
    df = pd.DataFrame(oof_rows, columns=['sample_id', 'repeat', 'fold',
                                         'y_true', 'y_pred'])
    vals = []
    for rep in range(5):
        sub = df[df['repeat'] == rep]
        vals.append(metric_block(sub['y_true'].values, sub['y_pred'].values)['r2'])
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def cluster_bootstrap_r2(oof_df, model_cols, n_boot=10000, seed=BOOT_SEED,
                         cluster_col='seed_roi'):
    """Vectorized seed-cluster paired bootstrap over an OOF frame with one
    row per (sample, repeat, fold, model). Returns per-contrast arrays.

    oof_df: columns sample_id, repeat, y_true + one y_pred per model (model_cols).
    cluster map (sample -> seed) must be joinable from ctx.
    """
    df = oof_df.merge(cluster_map(), on='sample_id', how='inner')
    g = df.groupby([cluster_col, 'repeat'])
    stats = {}
    for m in model_cols:
        y = g['y_true'].apply(np.asarray)
        p = g[m].apply(np.asarray)
        stats[m] = pd.DataFrame({
            'n': g.size(),
            'Sy': y.apply(np.nansum), 'Syy': y.apply(lambda v: np.nansum(v * v)),
            'Sp': p.apply(np.nansum), 'Spp': p.apply(lambda v: np.nansum(v * v)),
            'SSE': y.combine(p, lambda a, b: np.nansum((a - b) ** 2)),
        }).reset_index()
    clusters = sorted(set().union(*[set(s[cluster_col]) for s in stats.values()]))
    n_clu = len(clusters)
    clu_idx = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clu, size=(n_boot, n_clu))
    mults = np.zeros((n_boot, n_clu), dtype=np.int64)
    for b in range(n_boot):
        np.add.at(mults[b], draws[b], 1)
    boot = {m: np.full(n_boot, np.nan) for m in model_cols}
    for m in model_cols:
        s = stats[m].copy()
        s['ci'] = s[cluster_col].map(clu_idx).values
        s['ri'] = s['repeat'].values
        acc = np.zeros(n_boot)
        nrep = 0
        for rep in range(5):
            sub = s[s['ri'] == rep]
            if len(sub) == 0:
                continue
            sm = mults[:, sub['ci'].values]                       # (B, n)
            N = (sm * sub['n'].values[None, :]).sum(axis=1)
            Sy = (sm * sub['Sy'].values[None, :]).sum(axis=1)
            Syy = (sm * sub['Syy'].values[None, :]).sum(axis=1)
            SSE = (sm * sub['SSE'].values[None, :]).sum(axis=1)
            sst = Syy - Sy * Sy / np.maximum(N, 1)
            r2 = np.where(np.isfinite(sst) & (sst > 0),
                          1 - SSE / np.maximum(sst, 1e-300), np.nan)
            acc += np.where(np.isfinite(r2), r2, 0.0)
            nrep += 1
        boot[m] = acc / nrep
    return boot


def cluster_map():
    support = pd.read_csv(os.path.join(SMCR, 'confirmatory', 'support',
                                       'COMMON_SUPPORT.csv'))
    return support[['sample_id', 'seed_roi']].drop_duplicates('sample_id')


def _seed_derived(sid, tag):
    return int.from_bytes(hashlib.sha256(
        f'v1B5{tag}{sid}'.encode()).digest()[:8], 'big')


def c6c8r_from_paths(eng, path_lists, L):
    """C6/C8R process features for a cell from an arbitrary path list
    (list of node tuples). Returns (blind56, ob32_dict)."""
    from bridge_noscreen import blind_from_paths, ob_from_paths, METRICS
    paths = np.asarray(path_lists, dtype=np.int16)
    if len(paths) == 0:
        return np.full(56, np.nan), {f'ob_{m}_p{q}_{a}': np.nan
                                     for m in METRICS for q in range(4)
                                     for a in ('med', 'iqr')}
    return blind_from_paths(paths, L), ob_from_paths(paths, L)


def repeat_summary_from_oof_rows(rows_by_model):
    """rows_by_model: dict model -> list of (sample_id, repeat, fold, yt, yp).
    Returns per-model repeat-summary R2."""
    return {m: repeat_summary_r2(rows) for m, rows in rows_by_model.items()}
