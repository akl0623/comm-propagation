#!/usr/bin/env python3
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, 'cucr_secondary_analysis_v1')
sys.path.append(os.path.join(ROOT, 's10_optimized_v1', 'code'))
sys.path.append(os.path.join(ROOT, 'screen_match_confirm_v1',
                                'confirmatory', 'code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import so_lib as SO
import stage4a_lib as A4
from cucr_engine import CucrEngine, CucrMatrices
from cucr_s2_features import ob_from_paths, OB_COLS, N_C4, N_C8R

FEAT = os.path.join(OUT, 'features')
CV = os.path.join(OUT, 'cv')
RES = os.path.join(OUT, 'results')
REP = os.path.join(OUT, 'reports')
PATHLIB = os.path.join(OUT, 'path_library', 'CUCR_ALL_SAMPLED_PATHS.parquet')
os.makedirs(RES, exist_ok=True)

SCREEN_ID = 'S10'
N_SHUFFLE_REPS = 3


def log(msg):
    print(msg, flush=True)


def load_set(tag):
    X_c4 = np.load(os.path.join(FEAT, f'X_C4_{tag}.npy'))
    X_c8r = np.load(os.path.join(FEAT, f'X_C8R_{tag}.npy'))
    y = pd.read_parquet(os.path.join(FEAT, f'y_endpoint_{tag}.parquet'))
    meta = pd.read_parquet(os.path.join(FEAT, f'metadata_{tag}.parquet'))
    fm = pd.read_csv(os.path.join(CV, f'CUCR_{tag}_fold_manifest.csv'))
    sids = list(meta['sample_id'].values)
    sid_pos = {s: i for i, s in enumerate(sids)}
    return {'tag': tag, 'X_c4': X_c4, 'X_c8r': X_c8r,
            'y': y['y_endpoint'].values.astype(np.float64),
            'meta': meta, 'fm': fm, 'sids': sids, 'sid_pos': sid_pos}


def run_arm(setd, X, arm, fm=None):
    """Outer 5x5 CV for one arm; returns (oof_df, fold_metrics_df)."""
    fm = fm if fm is not None else setd['fm']
    y = setd['y']
    sid_pos = setd['sid_pos']
    oof_rows, fm_rows = [], []
    for rep in sorted(fm['repeat'].unique()):
        rd = fm[fm['repeat'] == rep]
        for fold in sorted(rd['test_fold'].unique()):
            te_sids = rd.loc[rd['test_fold'] == fold, 'sample_id'].tolist()
            tr_sids = rd.loc[rd['test_fold'] != fold, 'sample_id'].tolist()
            tr = np.array([sid_pos[s] for s in tr_sids], dtype=np.int64)
            te = np.array([sid_pos[s] for s in te_sids], dtype=np.int64)
            groups = rd.loc[rd['test_fold'] != fold, 'seed_roi'].values
            res = A4.train_outer_fold(X[tr], y[tr], groups, X[te])
            yt, yp = y[te], res['y_pred']
            mb = A4.metric_block(yt, yp)
            oof_rows += [{'sample_id': sid, 'arm': arm, 'repeat': int(rep),
                          'outer_fold': int(fold), 'y_true': float(yt[i]),
                          'y_pred': float(yp[i])}
                         for i, sid in enumerate(te_sids)]
            fm_rows.append({'arm': arm, 'repeat': int(rep),
                            'outer_fold': int(fold), **mb,
                            'best_alpha': res['best_alpha'],
                            'best_l1_ratio': res['best_l1_ratio'],
                            'best_inner_rmse': res['best_inner_rmse'],
                            'n_iter_': res['n_iter_'],
                            'n_warnings': res['n_warnings'],
                            'n_train': len(tr), 'n_test': len(te)})
    return pd.DataFrame(oof_rows), pd.DataFrame(fm_rows)


def run_set(setd, arms=('C4', 'C8R'), tag_prefix=None):
    """Run both arms on one support set; return dict of OOF/fold frames."""
    out = {}
    for arm in arms:
        X = setd['X_c4'] if arm == 'C4' else setd['X_c8r']
        oof, fmd = run_arm(setd, X, arm)
        tag = tag_prefix or setd['tag']
        oof.to_parquet(os.path.join(RES, f'OOF_{tag}__{arm}.parquet'),
                       index=False)
        fmd.to_csv(os.path.join(RES, f'FOLD_METRICS_{tag}__{arm}.csv'),
                   index=False)
        out[arm] = (oof, fmd)
        log('  %s/%s done' % (tag, arm))
    return out


def summarize(oof_a, oof_b, smap, name, do_perm=True, tag=None):
    """so_lib.contrast-equivalent summary using the frozen inference funcs."""
    row = SO.contrast(oof_a, oof_b, smap, name, do_perm=do_perm, tag=tag)
    return row


def main():
    eng = CucrEngine()
    eng.load()
    set_exact = load_set('exact')
    set_relaxed = load_set('relaxed')
    log('exact n=%d relaxed n=%d' % (len(set_exact['sids']),
                                     len(set_relaxed['sids'])))

    # ── primary: exact-rule C4 vs C8R ──────────────────────────────
    log('exact-rule arms...')
    run_set(set_exact)

    # ── exploratory: coverage-relaxed C4 vs C8R ───────────────────
    log('relaxed arms...')
    run_set(set_relaxed)

    # ── negative control: order-shuffle x3 on the relaxed set ─────
    log('order-shuffle controls (relaxed set)...')
    pl = pd.read_parquet(PATHLIB)
    r_sids = set(set_relaxed['sids'])
    all_paths = {}
    for sid, g in pl[pl['sample_id'].isin(r_sids)].groupby('sample_id',
                                                            sort=False):
        all_paths[sid] = [[int(x) for x in s.split('|')]
                          for s in g.sort_values('path_index')[
                              'roi_id_sequence']]
    ctx_min = {'eng': eng, 'screen_id': SCREEN_ID}
    shuffle_oofs = []
    shuffle_stats = {}
    for rep in range(1, N_SHUFFLE_REPS + 1):
        sh_paths, stats = SO.build_order_shuffle_all(ctx_min, all_paths, rep)
        shuffle_stats[rep] = stats
        # rebuild C8R features from the shuffled path sets (relaxed row order)
        X_sh = np.empty((len(set_relaxed['sids']), N_C8R), dtype=np.float64)
        X_sh[:, :N_C4] = set_relaxed['X_c4']
        for i, sid in enumerate(set_relaxed['sids']):
            pl_i = sh_paths[sid]
            arr = np.asarray(pl_i, dtype=np.int16)
            X_sh[i, N_C4:] = ob_from_paths(
                eng, arr, int(set_relaxed['meta']['path_length_edges'].iloc[i]))
        arm = f'C8R-OrderShuffle{rep}'
        oof, fmd = run_arm(set_relaxed, X_sh, arm)
        oof.to_parquet(os.path.join(RES, f'OOF_relaxed__{arm}.parquet'),
                       index=False)
        fmd.to_csv(os.path.join(RES, f'FOLD_METRICS_relaxed__{arm}.csv'),
                   index=False)
        shuffle_oofs.append(oof)
        log('  %s done' % arm)
    # pooled shuffle arm (mean of the three realisations — frozen convention)
    sh_pool = (pd.concat(shuffle_oofs)
               .groupby(['sample_id', 'repeat', 'outer_fold'])
               .agg(y_true=('y_true', 'first'), y_pred=('y_pred', 'mean'))
               .reset_index())
    sh_pool.to_parquet(os.path.join(RES, 'OOF_relaxed__C8R-OrderShufflePool.parquet'),
                       index=False)

    # ── inference ─────────────────────────────────────────────────
    smap_exact = set_exact['meta'][['sample_id', 'seed_roi']]
    smap_relaxed = set_relaxed['meta'][['sample_id', 'seed_roi']]
    oof_exact_c4 = pd.read_parquet(os.path.join(RES, 'OOF_exact__C4.parquet'))
    oof_exact_c8 = pd.read_parquet(os.path.join(RES, 'OOF_exact__C8R.parquet'))
    oof_rel_c4 = pd.read_parquet(os.path.join(RES, 'OOF_relaxed__C4.parquet'))
    oof_rel_c8 = pd.read_parquet(os.path.join(RES, 'OOF_relaxed__C8R.parquet'))

    rows = []
    rows.append(summarize(oof_exact_c8, oof_exact_c4, smap_exact,
                          'C8R-C4 exact', tag='cucr_v1_exact_C8R-C4'))
    rows.append(summarize(oof_rel_c8, oof_rel_c4, smap_relaxed,
                          'C8R-C4 relaxed', tag='cucr_v1_relaxed_C8R-C4'))
    rows.append(summarize(sh_pool, oof_rel_c4, smap_relaxed,
                          'C8R-ShufflePool-C4 relaxed',
                          tag='cucr_v1_relaxed_shufflepool-C4'))
    rows.append(summarize(oof_rel_c8, sh_pool, smap_relaxed,
                          'C8R-ShufflePool relaxed',
                          tag='cucr_v1_relaxed_C8R-shufflepool'))
    contrast_df = pd.DataFrame(rows)
    contrast_df.to_csv(os.path.join(RES, 'CUCR_CONTRASTS.csv'), index=False)

    for r in rows:
        log('%-28s r2a=%.4f r2b=%.4f d=%.4f CI=[%.4f, %.4f] p=%.4f rep_pos=%d/%d'
            % (r['contrast'], r['r2_a'], r['r2_b'], r['point_estimate'],
               r['bootstrap_ci_lo'], r['bootstrap_ci_hi'],
               r['permutation_p_two_sided'] or float('nan'),
               r['n_repeats_positive'], r['n_repeats']))

    # ── fold-metric summary ────────────────────────────────────────
    summary = {'protocol': {
        'estimator': 'ElasticNet(max_iter=200000, tol=1e-8, '
                     'selection=cyclic, random_state=42)',
        'grid': 'GRID120: alpha=logspace(-6,1,15) x l1_ratio='
                '[.05,.10,.25,.50,.75,.90,.95,1.0]',
        'inner_cv': 'GroupKFold(5) by seed ROI; min mean RMSE; tie -> '
                    'larger alpha, then larger l1_ratio',
        'outer_cv': '5 repeats x 5 folds, seed-ROI-grouped '
                    '(frozen grouped_fold_manifest, seeds '
                    '20260727-20260731)',
        'preprocessing': 'train-only median impute -> VarianceThreshold(0) '
                         '-> StandardScaler',
        'metric': 'mean of per-repeat pooled OOF R2 (frozen Task C '
                  'repeat_r2 convention)',
        'inference': 'paired seed-cluster bootstrap 10,000 (frozen '
                     'BOOT_SEED) + paired sign-flip permutation 1,000 '
                     'within (seed, repeat)',
        'grouping': 'stimulation-ROI-grouped (cohort-level; Stage 1 '
                    'decision: group-level universe carries no subject '
                    'dimension)',
        'calls': 'frozen A4.train_outer_fold / A4.metric_block / '
                 'so_lib.contrast executed directly',
    }, 'contrasts': rows, 'order_shuffle_stats': shuffle_stats}
    with open(os.path.join(RES, 'CUCR_EXPERIMENT_SUMMARY.json'), 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    # ── report ─────────────────────────────────────────────────────
    ftract = {'support': 1582, 'c4_r2': 0.4369, 'c8r_r2': 0.4580,
              'dr2': 0.0211, 'ci': '[+0.0024, +0.0421]', 'p': 0.0492,
              'role': 'Primary'}
    def row_of(name):
        r = next(x for x in rows if x['contrast'] == name)
        return r
    ex = row_of('C8R-C4 exact')
    rx = row_of('C8R-C4 relaxed')
    sh = row_of('C8R-ShufflePool-C4 relaxed')
    lines = [
        '# CUCR Stage 3 — C4 vs C8R 直接终点概率预测实验报告',
        '',
        '## 1. 协议一致性确认（与 F-TRACT 主实验完全相同）',
        '| 项 | CUCR 本实验 | F-TRACT 冻结管线 | 一致 |',
        '|----|-------------|------------------|------|',
        '| 估计器 | ElasticNet(max_iter=200000, tol=1e-8, cyclic, rs=42) | 同 | ✅（直接调用冻结 A4.train_outer_fold） |',
        '| 网格 | GRID120 = 15 alpha × 8 l1_ratio | 同 | ✅ |',
        '| 内层 CV | GroupKFold(5) by seed ROI，最小 mean RMSE，平局取更大 alpha | 同 | ✅ |',
        '| 外层 CV | 5×5 seed-grouped，冻结 grouped_fold_manifest | 同（CUCR 新 manifest，同函数） | ✅ |',
        '| 预处理 | 训练折-only median impute → VT(0) → scaler | 同 | ✅ |',
        '| 指标 | 每 repeat pooled OOF R² 的均值（repeat_r2） | 同 | ✅ |',
        '| 推断 | 配对 seed-cluster bootstrap 10k + 配对 sign-flip permutation 1k | 同 | ✅（冻结函数直接调用） |',
        '| 特征 | C4 24 维 / C8R 56 维 Legendre 有序基 | 同（差分验证逐位一致） | ✅ |',
        '',
        '## 2. CV 分组方式与理由',
        '分组变量：**stimulation-ROI**（cohort-level）。Stage 1 决策：',
        'universe 为 group-level（y_endpoint 来自 36 受试者汇总组矩阵、候选',
        '路径来自组引擎），cell 无受试者维度；subject-grouped CV 仅适用于旧的',
        '单观测路径 per-subject 设计。受试者支撑数以协变量保留。',
        '',
        '## 3. 结果汇总',
        '| 数据集 | 支持数 | C4 R² | C8R R² | ΔR² | 95% CI | p 值 | 定位 |',
        '|--------|--------|-------|--------|-----|--------|------|------|',
        '| F-TRACT (S10) | 1582 | 0.4369 | 0.4580 | +0.0211 | [+0.0024, +0.0421] | 0.0492 | Primary |',
        f'| CUCR (exact) | 881 | {ex["r2_b"]:.4f} | {ex["r2_a"]:.4f} | '
        f'{ex["point_estimate"]:+.4f} | [{ex["bootstrap_ci_lo"]:+.4f}, '
        f'{ex["bootstrap_ci_hi"]:+.4f}] | {ex["permutation_p_two_sided"]:.4f} '
        f'| Secondary (robustness) |',
        f'| CUCR (relaxed) | 871 | {rx["r2_b"]:.4f} | {rx["r2_a"]:.4f} | '
        f'{rx["point_estimate"]:+.4f} | [{rx["bootstrap_ci_lo"]:+.4f}, '
        f'{rx["bootstrap_ci_hi"]:+.4f}] | {rx["permutation_p_two_sided"]:.4f} '
        f'| Exploratory sensitivity |',
        f'| CUCR relaxed, order-shuffle control (C8R-Shuf−C4) | 871 | '
        f'{sh["r2_b"]:.4f} | {sh["r2_a"]:.4f} | {sh["point_estimate"]:+.4f} '
        f'| [{sh["bootstrap_ci_lo"]:+.4f}, {sh["bootstrap_ci_hi"]:+.4f}] | '
        f'{sh["permutation_p_two_sided"]:.4f} | Negative control |',
        '',
        f'重复方向（C8R−C4, exact）：{ex["n_repeats_positive"]}/'
        f'{ex["n_repeats"]} 个 repeat 为正。',
        '',
        '## 4. 结论与局限性',
        '见报告正文（生成后自动填入下方）。',
        '',
        '## 5. 产物（results/）',
        '- OOF_exact__{C4,C8R}.parquet、OOF_relaxed__{C4,C8R}.parquet',
        '- OOF_relaxed__C8R-OrderShuffle{1,2,3}.parquet、'
        'OOF_relaxed__C8R-OrderShufflePool.parquet',
        '- FOLD_METRICS_*.csv、CUCR_CONTRASTS.csv、CUCR_EXPERIMENT_SUMMARY.json',
    ]
    with open(os.path.join(REP, 'CUCR_MODEL_REPORT.md'), 'w') as f:
        f.write('\n'.join(lines))
    log('wrote CUCR_MODEL_REPORT.md')


if __name__ == '__main__':
    main()
