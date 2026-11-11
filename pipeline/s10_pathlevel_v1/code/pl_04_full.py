#!/usr/bin/env python3
import os
import sys
import json
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_lib as PL                                        # noqa: E402
import pl_variants as PV                                   # noqa: E402

N_WORKERS = 14
G = {}


def _task(args):
    variant, rep, fold = args
    ctx = G['ctx']
    tr, te, test_sids, groups = PL.fold_ids(ctx, rep, fold)
    oof, fm = PL.run_fold(G['X'][variant], ctx['y'], tr, te, groups,
                          test_sids, rep, fold, variant)
    return oof, fm


def main():
    t0 = time.time()
    PL.log('=' * 70)
    PL.log('Phase 1.4 — full 5x5 (confirmatory pass after smoke STOP)')
    PL.log('=' * 70)
    ctx = PL.load_context()
    G['ctx'] = ctx
    c4 = PL.base_c4(ctx)

    X = {}
    for v in PV.FULL_VARIANTS:
        t = time.time()
        X[v], npaths = PV.build_variant_X(ctx, v, c4)
        PL.log(f'  features {v:24s} {str(X[v].shape):12s} '
               f'mean paths/cell {npaths.mean():7.1f} ({time.time()-t:.0f}s)')
    G['X'] = X

    fold_parts, oof_parts = [], []
    for variant in PV.FULL_VARIANTS:
        oof_p = os.path.join(PL.RES, f'OOF_PREDICTIONS_{variant}.parquet')
        fm_p = os.path.join(PL.RES, f'FOLD_METRICS_{variant}.csv')
        if os.path.exists(oof_p) and os.path.exists(fm_p):
            oof_parts.append(pd.read_parquet(oof_p))
            fold_parts.append(pd.read_csv(fm_p))
            PL.log(f'  {variant}: checkpoint reused')
            continue
        tasks = [(variant, rep, fold) for rep in range(5) for fold in range(5)]
        t = time.time()
        oof_rows, fold_rows = [], []
        with mp.Pool(N_WORKERS) as pool:
            for oof, fm in pool.imap_unordered(_task, tasks):
                oof_rows.extend(oof)
                fold_rows.append(fm)
        odf = pd.DataFrame(oof_rows)
        fdf = pd.DataFrame(fold_rows).sort_values(['repeat', 'outer_fold'])
        odf.to_parquet(oof_p, index=False)
        fdf.to_csv(fm_p, index=False)
        oof_parts.append(odf)
        fold_parts.append(fdf)
        PL.log(f'  {variant}: 25 folds done ({time.time()-t:.0f}s)')

    fold_df = pd.concat(fold_parts, ignore_index=True)
    fold_df.to_csv(os.path.join(PL.RES, 'OUTER_FOLD_METRICS.csv'), index=False)

    # shuffle arms: pooled arm R2 = mean over the 3 realisation R2s
    overall = []
    for variant in PV.FULL_VARIANTS:
        sub = oof_parts[PV.FULL_VARIANTS.index(variant)]
        rep_r2 = [PL.A4.metric_block(
            g['y_true'].values, g['y_pred'].values)['r2']
            for _, g in sub.groupby('repeat')]
        rep_r2 = [v for v in rep_r2 if np.isfinite(v)]
        overall.append({'variant': variant, 'level': 'repeat_summary',
                        'r2': float(np.mean(rep_r2)),
                        'n_repeats_used': len(rep_r2),
                        'per_repeat_r2': json.dumps(rep_r2)})
        sample_sub = sub.groupby('sample_id')[['y_true', 'y_pred']].mean()
        overall.append({'variant': variant, 'level': 'sample_level_mean',
                        'r2': float(PL.A4.metric_block(
                            sample_sub['y_true'].values,
                            sample_sub['y_pred'].values)['r2'])})
    for name, arms in (('C8R-OrderShuffle', PV.SHUFFLE_ARMS),
                       ('C8R-EdgeShuffle', PV.EDGE_ARMS)):
        rows = [r for r in overall if r['variant'] in arms
                and r['level'] == 'repeat_summary']
        overall.append({'variant': name, 'level': 'repeat_summary',
                        'r2': float(np.mean([r['r2'] for r in rows])),
                        'n_repeats_used': len(rows),
                        'per_repeat_r2': json.dumps(
                            [r['r2'] for r in rows]),
                        'pooling': 'mean of per-repeat R2 across the 3 '
                                   'independent shuffle realisations'})
    ov = pd.DataFrame(overall)
    ov.to_csv(os.path.join(PL.RES, 'OVERALL_METRICS.csv'), index=False)

    # replication check of the C8R-All arm against the prior run -------------
    prior = pd.read_parquet(os.path.join(PL.PRIOR_TASKC,
                                         'OOF_PREDICTIONS.parquet'))
    mine = oof_parts[PV.FULL_VARIANTS.index('C8R-All')].rename(
        columns={'y_pred': 'c8r_pred'})
    m = prior.merge(mine, on=['sample_id', 'repeat', 'outer_fold'],
                    suffixes=('_prior', '_mine'))
    n = len(m)
    bit = bool(np.allclose(m['c8r_pred_prior'].values,
                           m['c8r_pred_mine'].values, rtol=0, atol=0)
               and n == 7910)
    rep = {
        'variant': 'C8R-All',
        'comparison': 'prior S10 full-experiment C8R OOF vs this run',
        'n_rows_compared': int(n),
        'bit_identical': bit,
        'prior_repeat_summary_r2': float(
            pd.read_csv(os.path.join(PL.PRIOR_TASKC,
                                     'OVERALL_METRICS.csv'))
            .query("model=='C8R' and level=='repeat_summary'")['r2'].iloc[0]),
        'this_repeat_summary_r2': float(
            ov.query("variant=='C8R-All' and level=='repeat_summary'"
                     )['r2'].iloc[0]),
    }
    with open(os.path.join(PL.RES, 'REPLICATION_CHECK.json'), 'w') as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    PL.log(f'\nreplication check C8R-All: bit_identical={bit} '
           f'(prior R2 {rep["prior_repeat_summary_r2"]:.4f} vs '
           f'{rep["this_repeat_summary_r2"]:.4f})')

    r2 = {r['variant']: r['r2'] for r in overall
          if r['level'] == 'repeat_summary'}
    PL.log('')
    PL.log('--- full 5x5 repeat-summary R2 ---')
    for v in ['C4', 'C8R-All', 'C8R-S10Path', 'C8R-OrderShuffle',
              'C8R-EdgeShuffle', 'C8R-MatchedRandom', 'C8R-SCNull',
              'C8R-LatencyNull', 'C6-All', 'C6-S10Path']:
        PL.log(f'  {v:22s} {r2.get(v, float("nan")):.4f}   '
               f'dR2 vs C4 = {r2.get(v, float("nan"))-r2.get("C4", float("nan")):+.4f}')
    PL.log(f'wall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
