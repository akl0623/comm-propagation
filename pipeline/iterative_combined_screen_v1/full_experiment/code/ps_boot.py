#!/usr/bin/env python3
import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ps_lib as P
from ps_lib import metric_block

warnings.filterwarnings('ignore')

B = 10000


def seed_cluster_bootstrap_paired(oof, models, seed_map, b, seed):
    df = oof.merge(seed_map, on='sample_id', how='inner')
    stats = {}
    for m in models:
        sub = df[df['model'] == m]
        gg = sub.groupby(['seed_roi', 'repeat'])
        y = gg['y_true'].apply(np.asarray)
        p = gg['y_pred'].apply(np.asarray)
        stats[m] = pd.DataFrame({
            'n': gg.size(), 'Sy': y.apply(np.nansum),
            'Syy': y.apply(lambda v: np.nansum(v * v)),
            'SSE': y.combine(p, lambda a, b: np.nansum((a - b) ** 2)),
        }).reset_index()
    clusters = sorted(set().union(*[set(s['seed_roi'])
                                    for s in stats.values()]))
    n_clu = len(clusters)
    clu_idx = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clu, size=(b, n_clu))
    mults = np.zeros((b, n_clu), dtype=np.int64)
    for j in range(b):
        np.add.at(mults[j], draws[j], 1)
    boot = {}
    for m in models:
        s = stats[m].copy()
        s['ci'] = s['seed_roi'].map(clu_idx).values
        s['ri'] = s['repeat'].values
        nrep = int(s['ri'].nunique())
        acc = np.zeros(b)
        n_finite = np.zeros(b)
        for rep in range(nrep):
            sub = s[s['ri'] == rep]
            if len(sub) == 0:
                continue
            sm = mults[:, sub['ci'].values]
            N = (sm * sub['n'].values[None, :]).sum(axis=1)
            Sy = (sm * sub['Sy'].values[None, :]).sum(axis=1)
            Syy = (sm * sub['Syy'].values[None, :]).sum(axis=1)
            SSE = (sm * sub['SSE'].values[None, :]).sum(axis=1)
            sst = Syy - Sy * Sy / np.maximum(N, 1)
            r2 = np.where(np.isfinite(sst) & (sst > 0),
                          1 - SSE / np.maximum(sst, 1e-300), np.nan)
            acc += np.where(np.isfinite(r2), r2, 0.0)
            n_finite += np.isfinite(r2)
        boot[m] = np.where(n_finite > 0, acc / np.maximum(n_finite, 1), np.nan)
    return boot


def two_sided_p(d):
    return 2 * min((d <= 0).mean(), (d >= 0).mean())


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    m = len(pvals)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, pvals[i] * (m - rank))
    for rank in range(1, m):
        adj[order[rank]] = max(adj[order[rank]], adj[order[rank - 1]])
    return adj


def bh(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    for rank, i in enumerate(order):
        adj[i] = min(1.0, pvals[i] * m / (rank + 1))
    for rank in range(m - 2, -1, -1):
        adj[order[rank]] = min(adj[order[rank]], adj[order[rank + 1]])
    return adj


def _point(oof, model):
    df = oof[oof['model'] == model]
    vals = [metric_block(sub['y_true'].values, sub['y_pred'].values)['r2']
            for _, sub in df.groupby('repeat')]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals))


def bootstrap_family(oof, models, contrasts, seed_map, family):
    boot = seed_cluster_bootstrap_paired(oof, models, seed_map, B, P.BOOT_SEED)
    rows = []
    for cname, a, b in contrasts:
        da = boot[a]
        db = boot[b]
        delta = da - db
        n_eff = int(np.isfinite(delta).sum())
        point = _point(oof, a) - _point(oof, b)
        ci = np.percentile(delta[np.isfinite(delta)], [2.5, 97.5])
        rows.append({'family': family, 'contrast': cname,
                     'point_estimate': point,
                     'bootstrap_median': float(
                         np.median(delta[np.isfinite(delta)])),
                     'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]),
                     'n_effective': n_eff,
                     'raw_p': two_sided_p(delta[np.isfinite(delta)])})
    return pd.DataFrame(rows), boot


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'{P.SCREEN_ID} 2.3/2.4 bootstrap + multiplicity')
    print('=' * 70, flush=True)
    seed_map = P.cluster_map()

    all_rows = []
    oof_tc = pd.read_parquet(os.path.join(P.TC, 'OOF_PREDICTIONS.parquet'))
    long_tc = oof_tc.melt(id_vars=['sample_id', 'repeat', 'outer_fold',
                                   'y_true'],
                          value_vars=['c4_pred', 'c6_pred', 'c8r_pred'],
                          var_name='model', value_name='y_pred')
    long_tc['model'] = long_tc['model'].str.replace('_pred', '').str.upper()
    fam_c, boot_c = bootstrap_family(
        long_tc, ['C4', 'C6', 'C8R'],
        [('C6-C4', 'C6', 'C4'), ('C8R-C4', 'C8R', 'C4')],
        seed_map, 'F1_TASK_C')
    all_rows.append(fam_c)
    print(fam_c.to_string(index=False), flush=True)

    boot_a = {}
    long_a_by_ds = {}
    units_by_ds = {}
    for ds in ('DATASET_1129', 'DATASET_1130'):
        units = pd.read_parquet(os.path.join(
            P.AGG, f'{ds}_UNIQUE_ENDPOINT_UNITS.parquet'))
        units['sample_id'] = (units['dataset'] + '|' + units['stim_region']
                              + '|' + units['current_end_region'] + '|'
                              + units['subpath_length'].astype(str))
        units = units.rename(columns={'subpath_length': 'path_length_edges'})
        units_by_ds[ds] = units
        ta_seed_map = P.taskA_cluster_map(units)
        oof_a = pd.read_parquet(os.path.join(P.TA,
                                             f'{ds}_OOF_PREDICTIONS.parquet'))
        oof_a['repeat'] = 0
        long_a = oof_a.melt(id_vars=['sample_id', 'repeat', 'outer_fold',
                                     'y_true'],
                            value_vars=['c4_pred', 'c6_pred', 'c8r_pred'],
                            var_name='model', value_name='y_pred')
        long_a['model'] = long_a['model'].str.replace('_pred',
                                                      '').str.upper()
        long_a_by_ds[ds] = long_a
        fam, ba = bootstrap_family(
            long_a, ['C4', 'C6', 'C8R'],
            [('C6-C4', 'C6', 'C4'), ('C8R-C4', 'C8R', 'C4')],
            ta_seed_map, f'F3_TASK_A_{ds}')
        all_rows.append(fam)
        boot_a[ds] = ba
        print(fam.to_string(index=False), flush=True)
    boot_df = pd.concat(all_rows, ignore_index=True)
    boot_df.to_csv(os.path.join(P.INF, 'BOOTSTRAP_CI.csv'), index=False)

    perlen_rows = []
    for ds in ('DATASET_1129', 'DATASET_1130'):
        units = units_by_ds[ds]
        long_a = long_a_by_ds[ds]
        seed_map_a = P.taskA_cluster_map(units)
        meta_len = units.set_index('sample_id')['path_length_edges']
        for L_ in sorted(units['path_length_edges'].unique()):
            sids_l = set(units[units['path_length_edges'] == L_]['sample_id'])
            n_seeds = units[units['path_length_edges'] == L_][
                'stim_region'].nunique()
            if len(sids_l) < 20 or n_seeds < 10:
                continue
            sub = long_a[long_a['sample_id'].isin(sids_l)]
            boot = seed_cluster_bootstrap_paired(
                sub, ['C4', 'C6', 'C8R'], seed_map_a, B, P.BOOT_SEED)
            for cname, a, b in (('C6-C4', 'C6', 'C4'),
                                ('C8R-C4', 'C8R', 'C4')):
                delta = (boot[a] - boot[b])
                fin = delta[np.isfinite(delta)]
                if len(fin) < B * 0.5:
                    continue
                point = _point(sub, a) - _point(sub, b)
                ci = np.percentile(fin, [2.5, 97.5])
                median = float(np.median(fin))
                width = float(ci[1] - ci[0])
                unstable = bool(width > 1.0
                                or abs(median - point) > 0.05)
                perlen_rows.append({
                    'dataset': ds, 'L': int(L_), 'contrast': cname,
                    'point_estimate': point, 'ci_lo': float(ci[0]),
                    'ci_hi': float(ci[1]), 'raw_p': two_sided_p(fin),
                    'n_cells': int(len(sids_l)), 'n_seeds': int(n_seeds),
                    'n_effective': int(np.isfinite(delta).sum()),
                    'stability_flag': 'UNSTABLE_SMALL_STRATUM' if unstable
                    else ''})
    perlen = pd.DataFrame(perlen_rows)
    if len(perlen):
        qvals = []
        for (ds_, c_), g in perlen.groupby(['dataset', 'contrast']):
            stable = g[g['stability_flag'] == ''].copy()
            if len(stable):
                stable['bh_q'] = bh(stable['raw_p'].tolist())
            else:
                stable['bh_q'] = np.nan
            g2 = g.merge(stable[['dataset', 'L', 'contrast', 'bh_q']],
                         on=['dataset', 'L', 'contrast'], how='left')
            qvals.append(g2)
        perlen = pd.concat(qvals).sort_values(
            ['dataset', 'L', 'contrast']).reset_index(drop=True)
        perlen.to_csv(os.path.join(P.TA, 'PER_LENGTH_CONTRASTS.csv'),
                      index=False)

    mult_rows = []
    f1 = boot_df[boot_df['family'] == 'F1_TASK_C']
    perm_res = os.path.join(P.INF, 'PERMUTATION_RESULTS.json')
    if os.path.exists(perm_res):
        pr = json.load(open(perm_res))
        maxT = {'C6-C4': pr['p_maxT_C6_minus_C4'],
                'C8R-C4': pr['p_maxT_C8R_minus_C4']}
        for _, r in f1.iterrows():
            mult_rows.append({'family': 'F1_TASK_C', 'contrast': r['contrast'],
                              'raw_p': r['raw_p'],
                              'adjusted_p': maxT[r['contrast']],
                              'method': 'maxT_permutation',
                              'reject_0p05': maxT[r['contrast']] < 0.05})
    p1 = f1['raw_p'].tolist()
    adj1 = holm(p1)
    for i, (_, r) in enumerate(f1.iterrows()):
        mult_rows.append({'family': 'F1_TASK_C', 'contrast': r['contrast'],
                          'raw_p': r['raw_p'], 'adjusted_p': adj1[i],
                          'method': 'holm_bootstrap',
                          'reject_0p05': adj1[i] < 0.05})
    f3 = boot_df[boot_df['family'].str.startswith('F3_TASK_A')]
    p3 = f3['raw_p'].tolist()
    adj3 = holm(p3)
    for i, (_, r) in enumerate(f3.iterrows()):
        mult_rows.append({'family': 'F3_TASK_A', 'contrast': r['contrast'],
                          'raw_p': r['raw_p'], 'adjusted_p': adj3[i],
                          'method': 'holm_bootstrap',
                          'reject_0p05': adj3[i] < 0.05})
    if len(perlen):
        for _, r in perlen.iterrows():
            if r['stability_flag']:
                mult_rows.append({'family': f'F_PERLEN_{r["dataset"]}_'
                                            f'{r["contrast"].replace("-", "_")}',
                                  'contrast': f'L={int(r["L"])} '
                                              f'{r["contrast"]}',
                                  'raw_p': r['raw_p'],
                                  'adjusted_p': np.nan,
                                  'method': 'bh_fdr_UNSTABLE_EXCLUDED',
                                  'reject_0p05': False})
            else:
                mult_rows.append({'family': f'F_PERLEN_{r["dataset"]}_'
                                            f'{r["contrast"].replace("-", "_")}',
                                  'contrast': f'L={int(r["L"])} '
                                              f'{r["contrast"]}',
                                  'raw_p': r['raw_p'],
                                  'adjusted_p': r['bh_q'],
                                  'method': 'bh_fdr',
                                  'reject_0p05': r['bh_q'] < 0.05})
    mult = pd.DataFrame(mult_rows)
    mult.to_csv(os.path.join(P.INF, 'MULTIPLICITY.csv'), index=False)
    print(mult.to_string(index=False), flush=True)
    print(f'DONE ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
