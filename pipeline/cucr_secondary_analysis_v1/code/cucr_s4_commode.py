#!/usr/bin/env python3
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, 'cucr_secondary_analysis_v1')
sys.path.append(os.path.join(OUT, 'code'))
from cucr_engine import CucrEngine, METRICS  # noqa: E402

RAWP = os.path.join(ROOT, 'raw_inputs')
PATHLIB = os.path.join(OUT, 'path_library', 'CUCR_ALL_SAMPLED_PATHS.parquet')
META_EXACT = os.path.join(OUT, 'features', 'metadata_exact.parquet')
NER_PROB_P = os.path.join(os.environ.get('CUCR_MATRIX_DIR', os.path.dirname(ROOT)), 'response_probability_matrix_251114.txt')
FT_SUM_FULL = os.path.join(ROOT, 'expA_revised_v1', '03_contribution',
                           'SUMMARY_BY_LENGTH_FULLCOEF.csv')
FT_SUM_MED = os.path.join(ROOT, 'expA_revised_v1', '03_contribution',
                          'SUMMARY_BY_LENGTH_MEDIANCOEF.csv')
FT_TRANS_FULL = os.path.join(ROOT, 'expA_validation_v1', '02_mediancoef',
                             'transition_by_length_fullcoef.csv')
FT_TRANS_MED = os.path.join(ROOT, 'expA_validation_v1', '02_mediancoef',
                            'transition_by_length_mediancoef.csv')

CM = os.path.join(OUT, 'communication_patterns')
SUB = os.path.join(CM, '01_subpaths')
MODELS = os.path.join(CM, '02_models')
CONTRIB = os.path.join(CM, '03_contribution')
REP = os.path.join(CM, 'reporting')
for d in (SUB, MODELS, CONTRIB, REP):
    os.makedirs(d, exist_ok=True)

FEATS = ['nav_eff', 'rout_eff', 'search_info', 'communicability']
SHORT = ['nav', 'rout', 'search', 'comm']
ROUTING_SET = {'nav', 'rout'}
DIFFUSION_SET = {'search', 'comm'}
SOURCES = ['FULLCOEF', 'MEDIANCOEF']
MIN_SAMPLES, MIN_STIM, N_FOLDS = 50, 5, 5
ALPHAS = np.logspace(-6, 1, 15)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
GRID = {'alpha': ALPHAS, 'l1_ratio': L1_RATIOS}
EN_KW = dict(max_iter=200000, tol=1e-8, random_state=42)
ZERO_TOL = 1e-12
ACCUM_LS = (6, 10, 14)


def log(msg):
    print(msg, flush=True)


def dominant(coefs, zfeat):
    c = np.abs(coefs[None, :] * zfeat)
    all_zero = np.all(c < 1e-300, axis=1)
    best = np.argmax(c, axis=1)
    return c, best, all_zero


def make_search(n_groups_train):
    k = min(N_FOLDS, int(n_groups_train))
    return GridSearchCV(ElasticNet(**EN_KW), GRID, cv=GroupKFold(n_splits=k),
                        scoring='neg_mean_squared_error', n_jobs=8), k


def main():
    t_all = time.time()
    eng = CucrEngine()
    eng.load()
    mats = {k: eng.mats[m] for k, m in
            zip(FEATS, ['nav', 'rout', 'search', 'comm'])}
    prob = np.loadtxt(NER_PROB_P, dtype=np.float64)
    prob[prob < 0] = np.nan
    roi = eng.roi_order
    r2i = eng.r2i

    # ── 1. subpath table (exact S10 support cells) ──────────────────
    meta = pd.read_parquet(META_EXACT)
    support_sids = set(meta['sample_id'])
    pl = pd.read_parquet(PATHLIB, columns=['sample_id', 'path_index', 'L',
                                           'roi_id_sequence'])
    pl = pl[pl['sample_id'].isin(support_sids)].reset_index(drop=True)
    seqs = pl['roi_id_sequence'].str.split('|').map(
        lambda s: np.array(s, dtype=np.int64))
    L_arr = pl['L'].to_numpy()
    n_paths = len(pl)
    log('exact-support paths: %d' % n_paths)

    stim_l, ep_l, len_l, row_l = [], [], [], []
    for i in range(n_paths):
        s = seqs[i]
        m = s.size - 1
        stim_l.append(np.full(m, s[0], dtype=np.int64))
        ep_l.append(s[1:].astype(np.int64))
        len_l.append(np.arange(1, m + 1, dtype=np.int32))
        row_l.append(np.full(m, i, dtype=np.int64))
    stim = np.concatenate(stim_l)
    ep = np.concatenate(ep_l)
    subL = np.concatenate(len_l)
    row_idx = np.concatenate(row_l)
    del stim_l, ep_l, len_l, row_l

    target = prob[stim, ep]
    feat = {k: mats[k][stim, ep] for k in FEATS}
    mask_cp = np.isfinite(target)
    mask_feat = np.ones(stim.size, dtype=bool)
    for k in FEATS:
        mask_feat &= np.isfinite(feat[k])
    keep = mask_cp & mask_feat
    out = pd.DataFrame({
        'subpath_length': subL[keep],
        'stim_id': stim[keep], 'endpoint_id': ep[keep],
        'path_id': (pl['sample_id'].to_numpy()[row_idx] + '#' +
                    pl['path_index'].astype(str).to_numpy()[row_idx])[keep],
        'cell_id': pl['sample_id'].to_numpy()[row_idx][keep],
        'cond_prob': target[keep],
    })
    for k in FEATS:
        out[k] = feat[k][keep]
    out['stim_region'] = out['stim_id'].map(
        lambda i: roi[int(i)])
    out['endpoint_region'] = out['endpoint_id'].map(
        lambda i: roi[int(i)])
    out.to_parquet(os.path.join(SUB, 'cucr_subpath_samples.parquet'),
                   index=False)
    log('subpath rows kept: %d / %d (drop %d)'
        % (len(out), stim.size, int((~keep).sum())))

    # ── 2. per-length models ────────────────────────────────────────
    rows, status = [], {}
    for L, g in out.groupby('subpath_length', sort=True):
        n, n_stim = len(g), g['stim_region'].nunique()
        n_pairs = int(pd.unique(g['stim_id'].to_numpy() * 1000
                                + g['endpoint_id'].to_numpy()).size)
        if n < MIN_SAMPLES or n_stim < MIN_STIM:
            status[int(L)] = {'trained': False, 'n_samples': n,
                              'n_stim': n_stim,
                              'reason': ('n_samples<50' if n < MIN_SAMPLES
                                         else 'n_stim<5')}
            log('[L=%d] SKIP n=%d n_stim=%d' % (L, n, n_stim))
            continue
        t0 = time.time()
        X = g[FEATS].to_numpy(np.float64)
        y = g['cond_prob'].to_numpy(np.float64)
        grp = g['stim_region'].to_numpy()
        ld = os.path.join(MODELS, 'length_%d' % int(L))
        os.makedirs(ld, exist_ok=True)

        gkf = GroupKFold(n_splits=N_FOLDS)
        oof = np.full(n, np.nan)
        fold_id = np.full(n, -1, dtype=int)
        fold_coefs, fold_rows = [], []
        for k, (tr, te) in enumerate(gkf.split(X, y, grp), start=1):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            search, inner_k = make_search(pd.unique(grp[tr]).size)
            search.fit(Xtr, y[tr], groups=grp[tr])
            best = search.best_estimator_
            pred = best.predict(Xte)
            oof[te] = pred
            fold_id[te] = k
            fold_coefs.append(best.coef_.copy())
            fold_rows.append({
                'fold': k, 'n_train': len(tr), 'n_test': len(te),
                'n_stim_train': int(pd.unique(grp[tr]).size),
                'n_stim_test': int(pd.unique(grp[te]).size),
                'inner_cv_splits': inner_k,
                'best_alpha': best.alpha, 'best_l1_ratio': best.l1_ratio,
                'test_r2': r2_score(y[te], pred),
                'test_rmse': float(np.sqrt(mean_squared_error(y[te], pred))),
                'test_mae': float(mean_absolute_error(y[te], pred)),
                'intercept': float(best.intercept_),
                **{f'coef_{s}': float(c) for s, c in zip(SHORT, best.coef_)},
            })

        oof_r2 = float(r2_score(y, oof))
        FC = np.vstack(fold_coefs)
        med_coefs = np.median(FC, axis=0)

        sc_full = StandardScaler().fit(X)
        Xf = sc_full.transform(X)
        search_full, inner_k_full = make_search(pd.unique(grp).size)
        search_full.fit(Xf, y, groups=grp)
        bp = search_full.best_params_
        model_full = ElasticNet(**bp, **EN_KW).fit(Xf, y)
        full_coefs = model_full.coef_.copy()
        insample_r2 = float(r2_score(y, model_full.predict(Xf)))
        all_zero = bool(np.all(np.abs(full_coefs) < ZERO_TOL))

        pd.DataFrame({
            'stim_region': g['stim_region'].to_numpy(),
            'endpoint_region': g['endpoint_region'].to_numpy(),
            'path_id': g['path_id'].to_numpy(),
            'cell_id': g['cell_id'].to_numpy(),
            'actual': y, 'predicted': oof, 'fold': fold_id,
        }).to_csv(os.path.join(ld, 'oof_predictions.csv'), index=False)
        fm = pd.DataFrame(fold_rows)
        fm.to_csv(os.path.join(ld, 'fold_metrics.csv'), index=False)
        fm[['fold', 'best_alpha', 'best_l1_ratio', 'intercept']
           + [f'coef_{s}' for s in SHORT]].to_csv(
            os.path.join(ld, 'fold_coefficients.csv'), index=False)
        with open(os.path.join(ld, 'full_model.json'), 'w') as f:
            json.dump({
                'length': int(L), 'n_samples': n, 'n_stim': n_stim,
                'n_unique_pairs': n_pairs,
                'role': 'INTERPRETATION ONLY (expA mirror)',
                'best_params': {k_: float(v_) for k_, v_ in bp.items()},
                'full_data_gridsearch_inner_splits': inner_k_full,
                'intercept': float(model_full.intercept_),
                'coefs': {s: float(c) for s, c in zip(SHORT, full_coefs)},
                'coef_order': SHORT, 'feature_columns': FEATS,
                'scaler_mean': {s: float(m_) for s, m_ in
                                zip(SHORT, sc_full.mean_)},
                'scaler_scale': {s: float(v_) for s, v_ in
                                 zip(SHORT, sc_full.scale_)},
                'median_fold_coefs': {s: float(c) for s, c in
                                      zip(SHORT, med_coefs)},
                'in_sample_r2_DO_NOT_REPORT': insample_r2,
                'oof_r2': oof_r2,
                'all_coefs_zero': all_zero,
            }, f, indent=2)
        rows.append({'length': int(L), 'n_samples': n, 'n_stim': n_stim,
                     'n_unique_pairs': n_pairs,
                     'duplication_factor': round(n / n_pairs, 2),
                     'oof_r2': oof_r2,
                     **{f'full_coef_{s}': float(c) for s, c in
                        zip(SHORT, full_coefs)},
                     **{f'median_coef_{s}': float(c) for s, c in
                        zip(SHORT, med_coefs)},
                     'full_best_alpha': float(bp['alpha']),
                     'full_best_l1_ratio': float(bp['l1_ratio']),
                     'in_sample_r2_reference_only': insample_r2,
                     'note': ('NO_SIGNAL: all full-data coefficients are zero'
                              if all_zero else '')
                     + ('; OOF_R2_NEGATIVE' if oof_r2 < 0 else '')})
        status[int(L)] = {'trained': True, 'n_samples': n, 'n_stim': n_stim,
                          'n_unique_pairs': n_pairs, 'oof_r2': oof_r2,
                          'seconds': round(time.time() - t0, 1)}
        log('[L=%2d] n=%d stim=%d pairs=%d | OOF R2=%+.4f (in %.0fs)'
            % (L, n, n_stim, n_pairs, oof_r2, time.time() - t0))
    allm = pd.DataFrame(rows).sort_values('length')
    allm.to_csv(os.path.join(MODELS, 'ALL_LENGTHS_METRICS.csv'), index=False)
    with open(os.path.join(MODELS, 'GROUP_STATUS.json'), 'w') as f:
        json.dump({'status_by_length': status,
                   'trained': sorted(k for k, v in status.items()
                                     if v['trained']),
                   'skipped': sorted(k for k, v in status.items()
                                     if not v['trained'])}, f, indent=2)

    # ── 3. contributions ───────────────────────────────────────────
    frames = []
    sum_full, sum_med = [], []
    for L in sorted(k for k, v in status.items() if v['trained']):
        j = json.load(open(os.path.join(MODELS, 'length_%d' % L,
                                        'full_model.json')))
        coef_full = np.array([j['coefs'][s] for s in SHORT])
        coef_med = np.array([j['median_fold_coefs'][s] for s in SHORT])
        mean = np.array([j['scaler_mean'][s] for s in SHORT])
        scale = np.array([j['scaler_scale'][s] for s in SHORT])
        assert np.all(scale > 0), 'L=%d zero-variance feature' % L
        g = pd.read_parquet(os.path.join(SUB, 'cucr_subpath_samples.parquet'),
                            columns=['stim_region', 'endpoint_region',
                                     'cell_id', 'path_id', 'stim_id',
                                     'endpoint_id', 'subpath_length'] + FEATS,
                            filters=[('subpath_length', '==', L)])
        X = g[FEATS].to_numpy(np.float64)
        Z = (X - mean) / scale
        base = pd.DataFrame({
            'length': L,
            'stim_region': g.stim_region.to_numpy(),
            'endpoint_region': g.endpoint_region.to_numpy(),
            'cell_id': g.cell_id.to_numpy(),
            'path_id': g.path_id.to_numpy(),
            'stim_id': g.stim_id.to_numpy(),
            'endpoint_id': g.endpoint_id.to_numpy(),
        })
        for src, coefs in zip(SOURCES, (coef_full, coef_med)):
            c, best, all_zero = dominant(coefs, Z)
            src_df = base.copy()
            src_df['coef_source'] = src
            for k_, s in enumerate(SHORT):
                src_df['contrib_%s' % s] = c[:, k_]
            src_df['dominant_feature'] = np.where(
                all_zero, 'all_zero', np.array(SHORT)[best])
            src_df['dominant_type'] = np.where(
                all_zero, 'all_zero',
                np.where(src_df.dominant_feature.isin(ROUTING_SET),
                         'routing', 'diffusion'))
            src_df['coef_zero_count'] = int((np.abs(coefs) < 1e-12).sum())
            frames.append(src_df)
            valid = src_df[src_df.dominant_type != 'all_zero']
            feats_sel = np.array(SHORT)[best[~all_zero]]
            cm = c[~all_zero]
            total = float(cm.sum()) if cm.size else np.nan
            routing_share = float(
                cm[:, [SHORT.index(s) for s in ('nav', 'rout')]].sum()
                / total) if cm.size and total > 0 else np.nan
            (sum_full if src == 'FULLCOEF' else sum_med).append({
                'length': L, 'n_samples': len(src_df),
                'n_unique_pairs': int(pd.unique(
                    g.stim_id.to_numpy() * 1000
                    + g.endpoint_id.to_numpy()).size),
                'n_valid': len(valid),
                'n_all_zero': int((src_df.dominant_type == 'all_zero').sum()),
                'routing_count': int((valid.dominant_type == 'routing').sum()),
                'diffusion_count': int(
                    (valid.dominant_type == 'diffusion').sum()),
                'n_nav': int((feats_sel == 'nav').sum()),
                'n_rout': int((feats_sel == 'rout').sum()),
                'n_search': int((feats_sel == 'search').sum()),
                'n_comm': int((feats_sel == 'comm').sum()),
                'routing_contrib_share': routing_share,
                'diffusion_contrib_share': (1 - routing_share)
                if routing_share is not None else np.nan,
                'oof_r2': status[L]['oof_r2'],
            })
    allc = pd.concat(frames, ignore_index=True)
    allc.to_parquet(os.path.join(CONTRIB, 'cucr_sample_level_contributions.parquet'),
                    index=False)

    def summary_frame(rows_, src):
        df = pd.DataFrame(rows_)
        return pd.DataFrame({
            'length': df.length.astype(int), 'coef_source': src,
            'n_samples': df.n_samples, 'n_unique_pairs': df.n_unique_pairs,
            'n_all_zero_excluded': df.n_all_zero,
            'routing_ratio': df.routing_count / df.n_valid,
            'diffusion_ratio': df.diffusion_count / df.n_valid,
            'nav_ratio': df.n_nav / df.n_valid,
            'rout_ratio': df.n_rout / df.n_valid,
            'search_ratio': df.n_search / df.n_valid,
            'comm_ratio': df.n_comm / df.n_valid,
            'routing_contrib_share': df.routing_contrib_share,
            'diffusion_contrib_share': df.diffusion_contrib_share,
            'oof_r2': df.oof_r2,
        }).sort_values('length')

    sfull = summary_frame(sum_full, 'FULLCOEF')
    smed = summary_frame(sum_med, 'MEDIANCOEF')
    sfull.to_csv(os.path.join(CONTRIB, 'CUCR_SUMMARY_BY_LENGTH_FULLCOEF.csv'),
                 index=False)
    smed.to_csv(os.path.join(CONTRIB, 'CUCR_SUMMARY_BY_LENGTH_MEDIANCOEF.csv'),
                index=False)

    # ── 4. path-level transition ───────────────────────────────────
    plL = pd.read_parquet(PATHLIB, columns=['sample_id', 'path_index', 'L'])
    plL = plL[plL['sample_id'].isin(support_sids)].reset_index(drop=True)
    plL['path_id'] = plL['sample_id'] + '#' + plL['path_index'].astype(str)
    parent_L = plL.set_index('path_id')['L']
    trans = {}
    for src in SOURCES:
        d = allc[allc.coef_source == src].copy()
        d = d[d['coef_zero_count'] != 4]
        assert set(d['dominant_type'].unique()) <= {'routing', 'diffusion'}
        d = d.merge(parent_L.rename('parent_path_L'), on='path_id',
                    how='left', validate='many_to_one')
        assert d['parent_path_L'].notna().all()
        g = d.groupby('path_id', sort=False)
        agg = g.agg(full_length=('parent_path_L', 'first'),
                    n_scored_subpaths=('length', 'size'),
                    max_scored_subpath_length=('length', 'max'),
                    n_diffusion_subpaths=('dominant_type',
                                          lambda s: int((s == 'diffusion').sum())),
                    cell_id=('cell_id', 'first')).reset_index()
        agg['diffusion_subpath_ratio'] = (agg['n_diffusion_subpaths']
                                          / agg['n_scored_subpaths'])
        agg['is_partial_diffusion'] = agg['n_diffusion_subpaths'] > 0
        agg['has_complete_coverage'] = (agg['full_length']
                                        == agg['max_scored_subpath_length'])
        assert int((agg['n_scored_subpaths'] == 0).sum()) == 0
        rows_ = []
        for L_, gL in agg.groupby('full_length'):
            rows_.append({'full_length': int(L_),
                          'n_paths': int(len(gL)),
                          'n_cells': int(gL['cell_id'].nunique()),
                          'coverage_complete': bool(
                              gL['has_complete_coverage'].all()),
                          'n_partial_diffusion':
                              int(gL['is_partial_diffusion'].sum()),
                          'partial_ratio': float(
                              gL['is_partial_diffusion'].mean()),
                          'n_persistent_routing':
                              int((~gL['is_partial_diffusion']).sum()),
                          'persistent_ratio': float(
                              1 - gL['is_partial_diffusion'].mean())})
        by = pd.DataFrame(rows_).sort_values('full_length')
        by.to_csv(os.path.join(CONTRIB,
                               'cucr_transition_by_length_%s.csv'
                               % src.lower()), index=False)
        trans[src] = by
    log('transition tables written')

    # ── 5. probability-accumulation check ──────────────────────────
    def pred(df, L):
        d = df.loc[1:L, 'diffusion_ratio'].dropna().values.copy()
        if len(d) < L:
            return np.nan
        return float(1 - np.prod(1 - d))

    accum_rows = []
    for L in ACCUM_LS:
        row = {'L': int(L)}
        for src, by in (('FULLCOEF', trans['FULLCOEF']),
                        ('MEDIANCOEF', trans['MEDIANCOEF'])):
            sf = sfull if src == 'FULLCOEF' else smed
            p = pred(sf, L)
            obs = (float(by.loc[by['full_length'] == L,
                                'partial_ratio'].iloc[0])
                   if (by['full_length'] == L).any() else np.nan)
            row['pred_%s' % src] = p
            row['obs_%s' % src] = obs
            row['deviation_%s' % src] = (abs(obs - p)
                                         if pd.notna(p) and pd.notna(obs)
                                         else np.nan)
        accum_rows.append(row)
    accum = pd.DataFrame(accum_rows)
    accum.to_csv(os.path.join(CONTRIB, 'CUCR_ACCUMULATION_CHECK.csv'),
                 index=False)
    with open(os.path.join(CONTRIB, 'CUCR_TRANSITION_CHECK.json'), 'w') as f:
        json.dump({
            'definition': ('transition = partial-diffusion ratio rises with '
                           'full path length (manuscript full-path '
                           'definition); accumulation = 1 - prod(1 - d_k)'),
            'accumulation_check': json.loads(accum.to_json()),
            'group_status': status,
        }, f, indent=2, default=str)

    # ── 6. F-TRACT comparison (read reference CSVs, nothing hand-typed)
    ft_sf = pd.read_csv(FT_SUM_FULL)
    ft_sm = pd.read_csv(FT_SUM_MED)
    ft_tf = pd.read_csv(FT_TRANS_FULL)
    ft_tm = pd.read_csv(FT_TRANS_MED)

    def ft_routing(L, df):
        if (df['length'] == L).any():
            return float(df.loc[df['length'] == L, 'routing_ratio'].iloc[0])
        return np.nan

    def ft_partial(L, df):
        if (df['full_length'] == L).any():
            return float(df.loc[df['full_length'] == L,
                                'partial_ratio'].iloc[0])
        return np.nan

    def cu_routing(L, df):
        if (df['length'] == L).any():
            return float(df.loc[df['length'] == L, 'routing_ratio'].iloc[0])
        return np.nan

    def cu_partial(L, df):
        if (df['full_length'] == L).any():
            return float(df.loc[df['full_length'] == L,
                                'partial_ratio'].iloc[0])
        return np.nan

    cmp_rows = []
    for L in (1, 6):
        cmp_rows.append({
            'metric': 'L=%d routing ratio' % L,
            'FT_FULL': ft_routing(L, ft_sf), 'CU_FULL': cu_routing(L, sfull),
            'FT_MED': ft_routing(L, ft_sm), 'CU_MED': cu_routing(L, smed)})
        cmp_rows.append({
            'metric': 'L=%d partial-diffusion' % L,
            'FT_FULL': ft_partial(L, ft_tf), 'CU_FULL': cu_partial(
                L, trans['FULLCOEF']),
            'FT_MED': ft_partial(L, ft_tm), 'CU_MED': cu_partial(
                L, trans['MEDIANCOEF'])})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(CM, 'CUCR_VS_FTRACT_COMMODE.csv'), index=False)

    # ── 7. report ──────────────────────────────────────────────────
    lines = [
        '# CUCR Stage 4 — 通信模式分析报告（路径比例与路由-扩散转换）',
        '',
        '本阶段用**直接终点响应概率目标**重算 CUCR 通信模式分布，替代旧的逐步',
        '乘积目标结果。管线为 F-TRACT expA 通信模式分析的逐字镜像（per-length '
        'ElasticNet + FULLCOEF/MEDIANCOEF + |coef×z-feature| 贡献分类 + 手稿'
        '完整路径转换定义 + 概率累积验证）。',
        '',
        '## 1. 系数来源说明（与任务描述的偏差，已记录）',
        '任务要求"从 Stage 3 C8R 模型获取系数"。F-TRACT 参考分析（本阶段对比',
        '对象）使用 **per-length 4-度量模型**（expA 管线：每个 L 一个 '
        'ElasticNet，特征=直达 (seed, endpoint) 对的 4 个通信度量，目标=直接',
        '终点概率）；Stage 3 的单一 56 维 C8R 模型系数无法按子路径长度分解。',
        '为保持与 F-TRACT 对比的协议同一性，本阶段采用 per-length 模型族',
        '（同一直接终点目标、同一 4 个共享度量矩阵、同一估计器族），并在',
        'exact S10 支持集（881 cells，严格规则支持充足，无需使用放宽集）上拟合。',
        '',
        '## 2. 逐长度模型状态',
        '| L | n | n_stim | n_pairs | OOF R² | 备注 |',
        '|-----|-----|--------|---------|--------|------|',
    ]
    for _, r in allm.iterrows():
        lines.append('| %d | %d | %d | %d | %+.4f | %s |'
                     % (r.length, r.n_samples, r.n_stim, r.n_unique_pairs,
                        r.oof_r2, r.note))
    lines += [
        '',
        '## 3. 子路径通信模式比例（routing vs diffusion，样本级）',
        '| L | n | FULLCOEF routing | FULLCOEF diffusion | MEDIANCOEF '
        'routing | MEDIANCOEF diffusion |',
        '|---|----|-----------------|--------------------|'
        '-------------------|----------------------|',
    ]
    for _, rf in sfull.iterrows():
        rm = smed[smed.length == rf.length].iloc[0]
        lines.append('| %d | %d | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'
                     % (rf.length, rf.n_samples, 100 * rf.routing_ratio,
                        100 * rf.diffusion_ratio, 100 * rm.routing_ratio,
                        100 * rm.diffusion_ratio))
    lines += [
        '',
        '## 4. 完整路径转换（partial-diffusion path 比例，手稿定义）',
        '| L | n_paths | FULLCOEF partial | MEDIANCOEF partial |',
        '|---|---------|------------------|--------------------|',
    ]
    for _, r in trans['FULLCOEF'].iterrows():
        rm = trans['MEDIANCOEF']
        pmed = (float(rm.loc[rm.full_length == r.full_length,
                             'partial_ratio'].iloc[0])
                if (rm.full_length == r.full_length).any() else np.nan)
        lines.append('| %d | %d | %.1f%% | %s |'
                     % (r.full_length, r.n_paths, 100 * r.partial_ratio,
                        '%.1f%%' % (100 * pmed) if pd.notna(pmed) else '—'))
    lines += [
        '',
        '## 5. 概率累积机制验证（expected = 1 − Π(1 − p_k)）',
        '| L | pred FULL | obs FULL | dev FULL | pred MED | obs MED | '
        'dev MED |',
        '|---|-----------|----------|----------|----------|---------|--------|',
    ]
    for _, r in accum.iterrows():
        lines.append('| %d | %s | %s | %s | %s | %s | %s |'
                     % (r.L,
                        '%.3f' % r.pred_FULLCOEF if pd.notna(r.pred_FULLCOEF)
                        else '—',
                        '%.3f' % r.obs_FULLCOEF if pd.notna(r.obs_FULLCOEF)
                        else '—',
                        '%.3f' % r.deviation_FULLCOEF
                        if pd.notna(r.deviation_FULLCOEF) else '—',
                        '%.3f' % r.pred_MEDIANCOEF
                        if pd.notna(r.pred_MEDIANCOEF) else '—',
                        '%.3f' % r.obs_MEDIANCOEF
                        if pd.notna(r.obs_MEDIANCOEF) else '—',
                        '%.3f' % r.deviation_MEDIANCOEF
                        if pd.notna(r.deviation_MEDIANCOEF) else '—'))
    lines += [
        '',
        '## 6. 与 F-TRACT 对比（数值来自冻结参考 CSV，未手抄）',
        '| 指标 | F-TRACT FULLCOEF | CUCR FULLCOEF | F-TRACT MEDIANCOEF | '
        'CUCR MEDIANCOEF |',
        '|------|------------------|---------------|--------------------|'
        '-----------------|',
    ]
    for _, r in cmp.iterrows():
        fmt = lambda v: ('%.1f%%' % (100 * v)) if pd.notna(v) else '—'
        lines.append('| %s | %s | %s | %s | %s |'
                     % (r.metric, fmt(r.FT_FULL), fmt(r.CU_FULL),
                        fmt(r.FT_MED), fmt(r.CU_MED)))
    lines += [
        '',
        '注：任务摘要中的"F-TRACT FULLCOEF routing ≥80.9% at L=1-17"与冻结',
        '参考表不符（实际 FULLCOEF routing_ratio 在 L=3 为 36.6%、L=6-7 约',
        '54-55%）；上表以冻结 CSV 实际值为准。',
        '',
        '## 7. 结论与局限性',
        '（占位，运行后填写）',
        '',
        '## 8. 产物',
        '- `communication_patterns/01_subpaths/cucr_subpath_samples.parquet`',
        '- `communication_patterns/02_models/length_*/` + `ALL_LENGTHS_METRICS.csv`',
        '- `communication_patterns/03_contribution/`（contributions、summary、'
        'transition、accumulation）',
        '- `CUCR_VS_FTRACT_COMMODE.csv`、本报告',
    ]
    with open(os.path.join(REP, 'CUCR_COMMODE_REPORT.md'), 'w') as f:
        f.write('\n'.join(lines))
    log('report written; total %.0fs' % (time.time() - t_all))


if __name__ == '__main__':
    main()
