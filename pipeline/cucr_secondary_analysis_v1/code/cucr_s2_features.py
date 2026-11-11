#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cucr_engine import (CucrEngine, feature_seed, K0_SAMPLE, METRICS)

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, 'cucr_secondary_analysis_v1')
UNI_P = os.path.join(OUT, 'universe', 'CUCR_OBSERVED_CELL_UNIVERSE.csv')
ATT_EXACT_P = os.path.join(OUT, 'screening', 'S10_EXACT_attrition.csv')
ATT_R1_P = os.path.join(OUT, 'screening', 'S10_R1_cucr_median_attrition.csv')
FEAT = os.path.join(OUT, 'features')
PATHLIB = os.path.join(OUT, 'path_library')
CV = os.path.join(OUT, 'cv')
REP = os.path.join(OUT, 'reports')
for d in (FEAT, PATHLIB, CV, REP):
    os.makedirs(d, exist_ok=True)

SCREEN_ID = 'S10'
OB_COLS = [f'ob_{m}_p{q}_{a}' for m in METRICS for q in range(4)
           for a in ('med', 'iqr')]
N_OB = len(OB_COLS)                      # 32
N_C4 = 20 + 4                            # 24
N_C8R = N_C4 + N_OB                      # 56

REPEAT_SEEDS = [20260727, 20260728, 20260729, 20260730, 20260731]  # frozen


def log(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════
# frozen math (verbatim)
def legendre_basis(L):
    """B[s, q] = P_q(z_s), z = -1 + 2(s-1)/(L-1) (L=1 -> z=0), q <= min(3, L-1)."""
    qmax = min(3, L - 1)
    if qmax < 0:
        return None
    z = np.array([0.0]) if L == 1 else np.asarray(
        -1.0 + 2.0 * np.arange(L) / (L - 1), dtype=np.float64)
    cols = [np.ones_like(z)]
    if qmax >= 1:
        cols.append(z)
    if qmax >= 2:
        cols.append((3.0 * z * z - 1.0) / 2.0)
    if qmax >= 3:
        cols.append((5.0 * z ** 3 - 3.0 * z) / 2.0)
    return np.column_stack(cols)


def ob_from_paths(eng, paths, L):
    """32 ordered-basis features — bridge_noscreen.ob_from_paths verbatim."""
    K = paths.shape[0]
    if K == 0:
        raise ValueError('empty path set for a formally sampled cell')
    u = paths[:, :-1]
    v = paths[:, 1:]
    B = legendre_basis(L)
    qmax = min(3, L - 1)
    out = {}
    for m in METRICS:
        X = eng.mats[m][u, v].astype(np.float64)
        beta = np.zeros((K, 4), dtype=np.float64)
        if qmax >= 0:
            Q, R = np.linalg.qr(B, mode='reduced')
            beta[:, :qmax + 1] = np.linalg.solve(R, Q.T @ X.T).T
        for q in range(4):
            col = beta[:, q]
            out[f'ob_{m}_p{q}_med'] = float(np.median(col))
            if K > 1:
                out[f'ob_{m}_p{q}_iqr'] = float(
                    np.subtract(*np.percentile(col, [75, 25])))
            else:
                out[f'ob_{m}_p{q}_iqr'] = 0.0
    return np.array([out[c] for c in OB_COLS], dtype=np.float64)


def base_c4(eng, uni):
    """so_lib.base_c4 verbatim: L one-hot (20) + 4 endpoint metrics."""
    L_arr = uni['path_length_edges'].values.astype(np.int64)
    onehot = np.zeros((len(uni), 20), dtype=np.float64)
    ok = L_arr <= 20
    onehot[ok] = np.eye(20)[L_arr[ok] - 1]
    si = np.array([eng.r2i[s] for s in uni['seed_roi']])
    ei = np.array([eng.r2i[e] for e in uni['endpoint_roi']])
    ep = np.stack([eng.mats[m][si, ei] for m in METRICS], axis=1)
    return np.hstack([onehot, ep])


def build_all_sampled(eng, uni):
    """so_lib.build_all_sampled verbatim (feature-path sampling)."""
    out = {}
    for _, r in uni.iterrows():
        sid = r['sample_id']
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        ranks = eng.sample_ranks(si, ei, L, K0_SAMPLE,
                                 feature_seed(SCREEN_ID, sid))
        if not ranks:
            out[sid] = []
            continue
        arr = eng.unrank_cell(si, ei, L, ranks)
        out[sid] = [[int(x) for x in row] for row in arr]
    return out


def grouped_fold_manifest(uni, n_folds=5):
    """so_lib.grouped_fold_manifest verbatim (5 repeats x seed-grouped folds)."""
    sizes = uni.groupby('seed_roi').size()
    rows = []
    for rep, rs in enumerate(REPEAT_SEEDS):
        seeds = np.array(sorted(sizes.index))
        np.random.default_rng(rs).shuffle(seeds)
        order = sorted(seeds, key=lambda s: (-int(sizes[s]),
                                             list(seeds).index(s)))
        load = np.zeros(n_folds, dtype=np.int64)
        assign = {}
        for s in order:
            f = int(np.argmin(load))
            assign[s] = f
            load[f] += int(sizes[s])
        for _, r in uni.iterrows():
            rows.append({'repeat': rep, 'repeat_seed': rs,
                         'sample_id': r['sample_id'],
                         'seed_roi': r['seed_roi'],
                         'test_fold': assign[r['seed_roi']]})
    return pd.DataFrame(rows)


def reject_reason(r, att):
    row = att[att['sample_id'] == r['sample_id']].iloc[0]
    if row['in_support']:
        return 'passed'
    if row['n_candidate'] == 0:
        return 'no_candidate_paths'
    need = 1 if row['L'] == 1 else 5
    if row['n_b5_valid'] < need:
        return 'insufficient_B5_valid_paths'
    return 'insufficient_variance_accepted_paths'


def save_set(tag, idx, eng, uni, X_c4, X_c8r, y, meta):
    """Save one support-set slice (parquet + npy)."""
    sub = uni.iloc[idx].reset_index(drop=True)
    np.save(os.path.join(FEAT, f'X_C4_{tag}.npy'), X_c4[idx])
    np.save(os.path.join(FEAT, f'X_C8R_{tag}.npy'), X_c8r[idx])
    pd.DataFrame(X_c4[idx],
                 index=sub['sample_id'].values).to_parquet(
        os.path.join(FEAT, f'X_C4_{tag}.parquet'))
    pd.DataFrame(X_c8r[idx],
                 index=sub['sample_id'].values).to_parquet(
        os.path.join(FEAT, f'X_C8R_{tag}.parquet'))
    pd.DataFrame({'sample_id': sub['sample_id'].values,
                  'y_endpoint': y[idx]}).to_parquet(
        os.path.join(FEAT, f'y_endpoint_{tag}.parquet'))
    meta.iloc[idx].reset_index(drop=True).to_parquet(
        os.path.join(FEAT, f'metadata_{tag}.parquet'))
    return sub


def main():
    eng = CucrEngine()
    eng.load()
    uni = pd.read_csv(UNI_P)
    att_exact = pd.read_csv(ATT_EXACT_P)
    att_r1 = pd.read_csv(ATT_R1_P)
    log('universe: %d cells' % len(uni))

    # ── 1. path library (frozen feature sampling) ──────────────────
    log('building all-sampled path library (K0=%d)...' % K0_SAMPLE)
    all_paths = build_all_sampled(eng, uni)
    rows = []
    for sid, pl in all_paths.items():
        for k, nodes in enumerate(pl):
            rows.append({'sample_id': sid, 'path_index': k,
                         'L': len(nodes) - 1,
                         'roi_id_sequence': '|'.join(str(int(x))
                                                     for x in nodes)})
    pd.DataFrame(rows).to_parquet(
        os.path.join(PATHLIB, 'CUCR_ALL_SAMPLED_PATHS.parquet'), index=False)
    log('path library: %d paths over %d cells'
        % (len(rows), len(all_paths)))

    # ── 2. features ────────────────────────────────────────────────
    log('computing C4 + C8R...')
    X_c4 = base_c4(eng, uni)
    X_c8r = np.empty((len(uni), N_C8R), dtype=np.float64)
    X_c8r[:, :N_C4] = X_c4
    n_paths = np.zeros(len(uni), dtype=np.int64)
    for i, (_, r) in enumerate(uni.iterrows()):
        pl = all_paths[r['sample_id']]
        n_paths[i] = len(pl)
        if not pl:
            X_c8r[i, N_C4:] = np.nan
            continue
        arr = np.asarray(pl, dtype=np.int16)
        X_c8r[i, N_C4:] = ob_from_paths(eng, arr, int(r['path_length_edges']))
    y = uni['y_endpoint'].values.astype(np.float64)

    # ── 3. metadata + support flags ────────────────────────────────
    uni['n_candidate'] = uni['T_candidate'].astype(int)
    uni['n_paths_sampled'] = n_paths
    uni['full_enumeration'] = uni['n_candidate'] <= K0_SAMPLE
    uni['in_support_exact'] = uni['sample_id'].map(
        dict(zip(att_exact['sample_id'], att_exact['in_support'])))
    uni['in_support_relaxed'] = uni['sample_id'].map(
        dict(zip(att_r1['sample_id'], att_r1['in_support'])))
    uni['reject_reason_exact'] = [
        reject_reason(r, att_exact) for _, r in uni.iterrows()]
    meta = uni[['sample_id', 'seed_roi', 'endpoint_roi', 'path_length_edges',
                'y_endpoint', 'n_candidate', 'n_paths_sampled',
                'full_enumeration', 'n_subjects_supporting',
                'in_support_exact', 'in_support_relaxed',
                'reject_reason_exact']]

    idx_full = np.arange(len(uni))
    idx_exact = np.where(uni['in_support_exact'].astype(bool).values)[0]
    idx_relaxed = np.where(uni['in_support_relaxed'].astype(bool).values)[0]
    save_set('full', idx_full, eng, uni, X_c4, X_c8r, y, meta)
    sub_exact = save_set('exact', idx_exact, eng, uni, X_c4, X_c8r, y, meta)
    sub_relaxed = save_set('relaxed', idx_relaxed, eng, uni, X_c4, X_c8r, y, meta)
    log('saved full=%d exact=%d relaxed=%d'
        % (len(idx_full), len(idx_exact), len(idx_relaxed)))

    # ── 4. fold manifest (frozen grouped_fold_manifest) ────────────
    fm_exact = grouped_fold_manifest(sub_exact)
    fm_exact.to_csv(os.path.join(CV, 'CUCR_exact_fold_manifest.csv'),
                    index=False)
    fm_relaxed = grouped_fold_manifest(sub_relaxed)
    fm_relaxed.to_csv(os.path.join(CV, 'CUCR_relaxed_fold_manifest.csv'),
                      index=False)

    # ── 5. quality checks ──────────────────────────────────────────
    qc = {}
    for tag, X, idx in (('full', X_c8r, idx_full),
                        ('exact', X_c8r, idx_exact),
                        ('relaxed', X_c8r, idx_relaxed)):
        Xs = X[idx]
        nan_cells = int(np.isnan(Xs).sum())
        inf_cells = int(np.isinf(Xs).sum())
        nan_rows = int(np.isnan(Xs).any(axis=1).sum())
        # zero-variance after median imputation (frozen pipeline order)
        X_imp = np.where(np.isnan(Xs), np.nanmedian(Xs, axis=0), Xs)
        zvar = np.where(np.var(X_imp, axis=0) < 1e-15)[0]
        y_s = y[idx]
        qc[tag] = {
            'n': int(len(idx)), 'shape': list(Xs.shape),
            'nan_cells': nan_cells, 'inf_cells': inf_cells,
            'nan_rows': nan_rows,
            'zero_var_cols_after_median_impute': [int(c) for c in zvar],
            'y_mean': float(y_s.mean()), 'y_sd': float(y_s.std()),
            'y_min': float(y_s.min()), 'y_max': float(y_s.max()),
            'y_zero_prop': float((y_s == 0).mean()),
            'n_paths_sampled': {
                'min': int(n_paths[idx].min()),
                'median': float(np.median(n_paths[idx])),
                'max': int(n_paths[idx].max()),
                'lt512': int((n_paths[idx] < 512).sum())},
        }
    # CV grouping stats per support set
    for tag, sub in (('exact', sub_exact), ('relaxed', sub_relaxed)):
        grp = sub.groupby('seed_roi').size()
        fm = fm_exact if tag == 'exact' else fm_relaxed
        test_sizes = fm.groupby(['repeat', 'test_fold']).size().describe()
        qc[tag]['cv_grouping'] = {
            'n_seed_rois': int(len(grp)),
            'cells_per_seed_min': int(grp.min()),
            'cells_per_seed_median': float(grp.median()),
            'cells_per_seed_max': int(grp.max()),
            'test_fold_sizes': {k: float(v) for k, v in
                                test_sizes[['min', '25%', '50%', '75%',
                                            'max']].to_dict().items()},
        }
    import json
    with open(os.path.join(OUT, 'audit', 'FEATURE_QC.json'), 'w') as f:
        json.dump(qc, f, indent=2, sort_keys=True, default=str)

    # ── 6. report ──────────────────────────────────────────────────
    L_exact = sub_exact['path_length_edges'].value_counts().sort_index()
    L_relaxed = sub_relaxed['path_length_edges'].value_counts().sort_index()
    lines = [
        '# CUCR Stage 2 — Feature Construction Report',
        '',
        '## 1. 概览',
        '| 指标 | 值 |',
        '|------|-----|',
        f'| 总单元数（universe，y>0） | {len(uni)} |',
        f'| 有完整路径库的单元数（T>0） | {int((n_paths > 0).sum())} |',
        f'| 路径不足 512 的单元数（全枚举，已标记） | {int((uni["n_candidate"] <= 512).sum())} |',
        f'| 严格 S10 支持数（_exact） | {len(idx_exact)} |',
        f'| Coverage-relaxed R1 支持数（_relaxed） | {len(idx_relaxed)} |',
        '',
        '## 2. 特征矩阵',
        '| 矩阵 | 维度 | 说明 |',
        '|------|------|------|',
        f'| X_C4 | {N_C4} | L one-hot(20) + 4 端点通信度量 (nav/rout/search/comm) |',
        f'| X_C8R | {N_C8R} | C4 + 32-dim Legendre 有序基 (4 度量 × P0-P3 × median/IQR) |',
        f'| y_endpoint | 1 | NER 组矩阵直接终点响应概率（逐步乘积目标已弃用） |',
        '',
        '## 3. 质量检查',
        '| 集合 | n | NaN cells | NaN rows | Inf | y mean±sd | y range | y=0 比例 |',
        '|------|---|-----------|----------|-----|-----------|---------|----------|',
    ]
    for tag in ('full', 'exact', 'relaxed'):
        q = qc[tag]
        lines.append(f'| {tag} | {q["n"]} | {q["nan_cells"]} | {q["nan_rows"]} '
                     f'| {q["inf_cells"]} | {q["y_mean"]:.3f}±{q["y_sd"]:.3f} '
                     f'| [{q["y_min"]:.3f}, {q["y_max"]:.3f}] '
                     f'| {q["y_zero_prop"]:.1%} |')
    lines += [
        '',
        'NaN 处理方式：仅 T=0 的 cell（无候选路径）的 C8R 32 维为 NaN '
        '（这些 cell 不在任何支持集中）；建模时按冻结管线 '
        'SimpleImputer(median) → VarianceThreshold(0) → StandardScaler 处理，'
        '全部在训练折上拟合。',
        '',
        '## 4. CV 分组（seed-ROI-grouped，冻结 grouped_fold_manifest）',
        '| 集合 | seed ROI 数 | 每 ROI cells (min/median/max) | 测试折大小 (min/25%/50%/75%/max) |',
        '|------|-------------|-------------------------------|----------------------------------|',
    ]
    for tag in ('exact', 'relaxed'):
        q = qc[tag]['cv_grouping']
        ts = q['test_fold_sizes']
        lines.append(f'| {tag} | {q["n_seed_rois"]} | '
                     f'{q["cells_per_seed_min"]}/{q["cells_per_seed_median"]:.0f}/'
                     f'{q["cells_per_seed_max"]} | '
                     f'{ts["min"]:.0f}/{ts["25%"]:.0f}/{ts["50%"]:.0f}/'
                     f'{ts["75%"]:.0f}/{ts["max"]:.0f} |')
    lines += [
        '',
        '## 5. 支持集 L 分布',
        f'_exact: {L_exact.to_dict()}',
        f'_relaxed: {L_relaxed.to_dict()}',
        '',
        '## 6. 拒绝原因（_exact）',
        '| 原因 | 单元数 |',
        '|------|--------|',
    ]
    for reason, n in meta['reject_reason_exact'].value_counts().items():
        lines.append(f'| {reason} | {n} |')
    lines += [
        '',
        '## 7. 产物',
        '- `features/X_C4_{full,exact,relaxed}.parquet` + `.npy`',
        '- `features/X_C8R_{full,exact,relaxed}.parquet` + `.npy`',
        '- `features/y_endpoint_{full,exact,relaxed}.parquet`',
        '- `features/metadata_{full,exact,relaxed}.parquet`',
        '- `path_library/CUCR_ALL_SAMPLED_PATHS.parquet`',
        '- `cv/CUCR_exact_fold_manifest.csv`, `cv/CUCR_relaxed_fold_manifest.csv`',
        '- `audit/FEATURE_QC.json`',
    ]
    with open(os.path.join(REP, 'CUCR_FEATURE_REPORT.md'), 'w') as f:
        f.write('\n'.join(lines))
    log('wrote CUCR_FEATURE_REPORT.md')


if __name__ == '__main__':
    main()
