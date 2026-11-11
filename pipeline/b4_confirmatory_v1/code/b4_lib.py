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
OUT = os.path.join(ROOT, 'b4_confirmatory_v1')
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.join(PACKAGE_ROOT, 'pipeline')
sys.path[:0] = [
    os.path.join(CODE_ROOT, 'b4_confirmatory_v1', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1',
                 'confirmatory', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1',
                 'candidate_screens', 'code'),
]

from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning

import stage4a_lib as A4
from stage4a_lib import (PreprocessingPipeline4A, metric_block,
                         ELASTICNET_PARAMS, build_engine)

warnings.filterwarnings('ignore')

TC = os.path.join(OUT, 'taskC')
TA = os.path.join(OUT, 'taskA')
INF = os.path.join(OUT, 'inference')
SUP = os.path.join(OUT, 'support')
NC = os.path.join(OUT, 'negative_controls')
STAB = os.path.join(OUT, 'stability')
VAL = os.path.join(OUT, 'validation')

MODELS = ['C4', 'C6', 'C8R']
BOOT_SEED = 8769577528861415417   # frozen Task C cluster-bootstrap convention


def b4_feature_seed(sid):
    return int.from_bytes(
        hashlib.sha256(f'v1B4{sid}'.encode()).digest()[:8], 'big')


def build_engine_b4():
    import bridge_noscreen as B
    eng = build_engine()
    B.ENGINE = eng
    A4.feature_cell_seed = b4_feature_seed
    return eng


def load_taskC_stage4a():
    """Returns (oof, fold_metrics, hp, support_meta, fm_sup, X, y, sid_pos)."""
    oof = pd.read_parquet(os.path.join(TC, 'OOF_PREDICTIONS.parquet'))
    fold_metrics = pd.read_csv(os.path.join(TC, 'OUTER_FOLD_METRICS.csv'))
    hp = pd.read_csv(os.path.join(TC, 'INNER_HYPERPARAMETERS.csv'))
    support = pd.read_csv(os.path.join(SUP, 'COMMON_SUPPORT.csv'))
    uni = pd.read_csv(A4.UNI_P)
    fm = pd.read_csv(A4.FOLD_P)
    support_sids = set(support[support['in_support']]['sample_id'])
    uni_sup = uni[uni['sample_id'].isin(support_sids)].reset_index(drop=True)
    uni_sup = uni_sup[['sample_id', 'seed_roi', 'endpoint_roi',
                       'path_length_edges']]
    eng = build_engine_b4()
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
            row = hp_m[(hp_m['repeat'] == rep)
                       & (hp_m['outer_fold'] == fold)]
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
        vals.append(metric_block(sub['y_true'].values,
                                 sub['y_pred'].values)['r2'])
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def cluster_map():
    support = pd.read_csv(os.path.join(SUP, 'COMMON_SUPPORT.csv'))
    return support[['sample_id', 'seed_roi']].drop_duplicates('sample_id')


def taskA_cluster_map(units):
    return units[['sample_id', 'stim_region']].rename(
        columns={'stim_region': 'seed_roi'}).drop_duplicates('sample_id')
