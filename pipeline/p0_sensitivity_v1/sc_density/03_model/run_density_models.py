#!/usr/bin/env python3
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

SMCR = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/screen_match_confirm_v1'
PLV = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/s10_pathlevel_v1'
SOV = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/s10_optimized_v1'
ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
for p in (os.path.join(SOV, 'code'),
          os.path.join(SMCR, 'confirmatory', 'code'),
          os.path.join(SMCR, 'candidate_screens', 'code'),
          os.path.join(PLV, 'code')):
    sys.path.insert(0, p)

import bridge_noscreen as B                                          # noqa: E402
import stage4a_lib as A4                                             # noqa: E402
import so_lib as SO                                                  # noqa: E402

P0 = os.path.join(ROOT, 'p0_sensitivity_v1/sc_density')
MATDIR = os.path.join(P0, '01_matrices')
COMMDIR = os.path.join(P0, '02_comm')
OUT = os.path.join(P0, '03_model')
CACHE = os.path.join(OUT, '_cache')
RAW = os.path.join(ROOT, 'raw_inputs')

DEFAULT_COMM = {
    'nav': os.path.join(RAW, 'navigation_efficiency_matrix_0.25density.csv'),
    'rout': os.path.join(RAW, 'rout_efficiency_matrix_0.25density.csv'),
    'search': os.path.join(RAW, 'search_information_matrix_0.25density.csv'),
    'comm': os.path.join(RAW, 'communicability_matrix_0.25density.csv'),
}
COMM_NAMES = {'nav': 'navigation_efficiency', 'rout': 'routing_efficiency',
              'search': 'search_information', 'comm': 'communicability'}

CONDITIONS = {
    # validation: exact main-analysis chain, no zeroing, main permutation tag
    'c0r2c': {'density': None, 'comm_dir': None, 'sc_zero': None,
              'zeroing': False, 'tag': 'H1|C8R_All_minus_C4',
              'label': '0.25 C0R2C reproduction check (no zeroing)'},
    'density_025': {'density': 0.25, 'comm_dir': None,
                    'sc_zero': 'tract_strength_density_025.csv',
                    'zeroing': True, 'tag': 'H1|C8R_All_minus_C4|density025',
                    'label': '0.25 historical chain + zeroing rule'},
    'density_020': {'density': 0.20, 'comm_dir': COMMDIR,
                    'sc_zero': 'tract_strength_density_020.csv',
                    'zeroing': True, 'tag': 'H1|C8R_All_minus_C4|density020',
                    'label': '0.20'},
    'density_030': {'density': 0.30, 'comm_dir': COMMDIR,
                    'sc_zero': 'tract_strength_density_030.csv',
                    'zeroing': True, 'tag': 'H1|C8R_All_minus_C4|density030',
                    'label': '0.30'},
}

N_WORKERS = 14
G = {}


# ══════════════════════════════════════════════════════════════════ engine
def load_engine_condition(cond):
    """Point bridge_noscreen at this condition's comm matrices and reset every
    engine singleton.  The engine itself is built inside SO.load_context (the
    one the feature builders actually read — stage4a_lib.build_engine always
    constructs a FRESH engine and reassigns B.ENGINE, so any engine built here
    would be discarded)."""
    B.NAV_P = DEFAULT_COMM['nav']
    B.ROUT_P = DEFAULT_COMM['rout']
    B.SRCH_P = DEFAULT_COMM['search']
    B.COMM_P = DEFAULT_COMM['comm']
    if cond['comm_dir']:
        tag = f"{cond['density']:.2f}".replace('.', '')
        B.NAV_P = os.path.join(COMMDIR, f'navigation_efficiency_density_{tag}.csv')
        B.ROUT_P = os.path.join(COMMDIR, f'routing_efficiency_density_{tag}.csv')
        B.SRCH_P = os.path.join(COMMDIR, f'search_information_density_{tag}.csv')
        B.COMM_P = os.path.join(COMMDIR, f'communicability_density_{tag}.csv')
    A4.ENGINE = None
    SO._ENG = None
    B.ENGINE = None


def zero_edges_by_sc(eng, strength_path):
    """Task-spec zeroing: per-edge metric = 0 where this density's SC is 0.
    Applied to eng.mats in place (only the C8R edge sequences read them)."""
    S = pd.read_csv(strength_path, header=None).values.astype(np.float64)
    mask = S <= 0
    n_zeroed_positions = int(mask.sum())
    for m in SO.METRICS:
        eng.mats[m][mask] = 0.0
    return n_zeroed_positions


# ══════════════════════════════════════════════════════════════════ features
def c8r_features(cond_key, cond, ctx, paths, c4):
    """C8R design matrix under this condition (bypasses so_lib's stratum cache,
    which must never be touched by sensitivity runs)."""
    cp = os.path.join(CACHE, f'X_C8R_{cond_key}.npy')
    npc = os.path.join(CACHE, f'npaths_C8R_{cond_key}.npy')
    if os.path.exists(cp):
        return np.load(cp), np.load(npc)
    X, n_paths = SO.build_c8r(ctx, paths, c4)
    np.save(cp, X)
    np.save(npc, n_paths)
    return X, n_paths


# ══════════════════════════════════════════════════════════════════ folds
def _task(args):
    arm, rep, fold = args
    ctx = G['ctx']
    tr, te, test_sids, groups = SO.fold_ids(ctx, rep, fold)
    return SO.run_fold(G['X'][arm], ctx['y'], tr, te, groups, test_sids,
                       rep, fold, arm)


def run_condition(cond_key, cond):
    t0 = time.time()
    outdir = os.path.join(OUT, cond_key)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    print(f'\n=== condition {cond_key}: {cond["label"]} ===', flush=True)

    load_engine_condition(cond)
    ctx = SO.load_context('H1')
    eng = ctx['eng']
    assert eng is B.ENGINE, 'engine singletons out of sync'
    G['ctx'] = ctx

    # fixed path set (main-analysis H1 all_sampled, 489,588 paths)
    paths = SO.load_path_set('H1', 'all_sampled')
    print(f'  paths: {sum(len(v) for v in paths.values())} across '
          f'{len(paths)} cells ({time.time()-t0:.0f}s)', flush=True)

    # C4 with RAW matrix values (main-analysis construction), then zero edges
    c4 = SO.base_c4(ctx)
    n_zeroed = 0
    if cond['zeroing']:
        n_zeroed = zero_edges_by_sc(
            eng, os.path.join(MATDIR, cond['sc_zero']))
        print(f'  zeroing rule applied: {n_zeroed} directed positions -> 0',
              flush=True)
    else:
        print('  zeroing rule: NOT applied (reproduction condition)', flush=True)

    # C8R features on the (possibly zeroed) matrices
    X_c8r, n_paths = c8r_features(cond_key, cond, ctx, paths, c4)
    G['X'] = {'C4': c4, 'C8R-All': X_c8r}
    print(f'  features: C4 {c4.shape}, C8R {X_c8r.shape}, '
          f'mean paths/cell {n_paths.mean():.1f} ({time.time()-t0:.0f}s)',
          flush=True)

    # 25 folds x 2 arms
    arms = ['C4', 'C8R-All']
    oof_parts, fold_parts = {}, []
    for arm in arms:
        tasks = [(arm, rep, fold) for rep in range(5) for fold in range(5)]
        oof_rows, fold_rows = [], []
        with mp.Pool(N_WORKERS) as pool:
            for oof, fm in pool.imap_unordered(_task, tasks):
                oof_rows.extend(oof)
                fold_rows.append(fm)
        odf = pd.DataFrame(oof_rows)
        fdf = pd.DataFrame(fold_rows).sort_values(['repeat', 'outer_fold'])
        name = 'C4' if arm == 'C4' else 'C8R'
        odf.to_parquet(os.path.join(outdir, f'OOF_{name}.parquet'), index=False)
        oof_parts[arm] = odf
        fold_parts.append(fdf)
        print(f'  {arm}: 25 folds done ({time.time()-t0:.0f}s)', flush=True)

    # OVERALL_METRICS (same format as the main run)
    overall = []
    for arm in arms:
        sub = oof_parts[arm]
        rep_r2 = [A4.metric_block(g['y_true'].values, g['y_pred'].values)['r2']
                  for _, g in sub.groupby('repeat')]
        rep_r2 = [v for v in rep_r2 if np.isfinite(v)]
        overall.append({'stratum': cond_key, 'arm': arm,
                        'level': 'repeat_summary',
                        'r2': float(np.mean(rep_r2)),
                        'n_repeats_used': len(rep_r2),
                        'per_repeat_r2': json.dumps(rep_r2)})
    pd.DataFrame(overall).to_csv(os.path.join(outdir, 'OVERALL_METRICS.csv'),
                                 index=False)

    # inference: paired bootstrap + sign-flip permutation
    smap = SO.seed_map(ctx)
    row = SO.contrast(oof_parts['C8R-All'], oof_parts['C4'], smap,
                      'C8R_All_minus_C4', do_perm=True, tag=cond['tag'])
    row['condition'] = cond_key
    print(f'  contrast: dR2={row["point_estimate"]:+.6f} '
          f'CI[{row["bootstrap_ci_lo"]:+.4f},{row["bootstrap_ci_hi"]:+.4f}] '
          f'p={row["permutation_p_two_sided"]:.4f} '
          f'repeats+ {row["n_repeats_positive"]}/{row["n_repeats"]} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return row, oof_parts


def reproduction_check(oof_parts):
    """c0r2c condition: bit-identity vs the published main-analysis OOF."""
    pub = os.path.join(SOV, 'strata', '_results')
    out = {}
    for arm, fname in (('C4', 'OOF_H1__C4.parquet'),
                       ('C8R-All', 'OOF_H1__C8R-All.parquet')):
        prior = pd.read_parquet(os.path.join(pub, fname))
        mine = oof_parts[arm].rename(columns={'y_pred': 'mine'})
        m = prior[['sample_id', 'repeat', 'outer_fold', 'y_pred']].merge(
            mine[['sample_id', 'repeat', 'outer_fold', 'mine']],
            on=['sample_id', 'repeat', 'outer_fold'], how='inner')
        bit = bool(len(m) == len(prior)
                   and np.array_equal(m['y_pred'].values, m['mine'].values))
        out[arm] = {'n_rows_compared': int(len(m)),
                    'bit_identical': bit,
                    'max_abs_diff': float(np.max(np.abs(
                        m['y_pred'].values - m['mine'].values))) if len(m)
                    else None}
        print(f'  reproduction {arm}: bit_identical={bit} '
              f'({out[arm]["n_rows_compared"]} rows)', flush=True)
    pd.DataFrame(out).T.reset_index().rename(columns={'index': 'arm'}).to_csv(
        os.path.join(OUT, 'c0r2c', 'REPRODUCTION_CHECK.csv'), index=False)
    return out


def main():
    targets = sys.argv[1:] or list(CONDITIONS)
    rows = []
    for cond_key in targets:
        cond = CONDITIONS[cond_key]
        row, oof_parts = run_condition(cond_key, cond)
        rows.append(row)
        if cond_key == 'c0r2c':
            reproduction_check(oof_parts)

    cols = ['condition', 'r2_a', 'r2_b', 'point_estimate', 'bootstrap_ci_lo',
            'bootstrap_ci_hi', 'bootstrap_frac_gt_0', 'n_boot_finite',
            'n_repeats_positive', 'n_repeats', 'repeats',
            'permutation_p_two_sided', 'perm_obs']
    pd.DataFrame(rows)[cols].to_csv(
        os.path.join(OUT, 'ALL_CONDITIONS_CONTRAST.csv'), index=False)
    print('\n=== ALL CONDITIONS ===', flush=True)
    for r in rows:
        print(f'  {r["condition"]:14s} dR2={r["point_estimate"]:+.6f}  '
              f'CI[{r["bootstrap_ci_lo"]:+.4f},{r["bootstrap_ci_hi"]:+.4f}]  '
              f'p={r["permutation_p_two_sided"]:.4f}  '
              f'repeats+ {r["n_repeats_positive"]}/{r["n_repeats"]}', flush=True)


if __name__ == '__main__':
    main()
