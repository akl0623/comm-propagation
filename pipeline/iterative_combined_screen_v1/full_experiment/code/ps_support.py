#!/usr/bin/env python3
import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bridge_noscreen as B
import ps_lib as P


def main():
    uni = pd.read_csv(B.UNI_P)
    fm = pd.read_csv(B.FOLD_P)
    att = pd.read_csv(P.ATTRITION)
    assert len(uni) == 2636 and len(att) == 2636

    support_sids = set(att[att['in_support']]['sample_id'])

    cs = uni[['sample_id', 'seed_roi', 'endpoint_roi',
              'path_length_edges']].copy()
    cs['in_support'] = cs['sample_id'].isin(support_sids)
    cs['n_paths'] = cs['sample_id'].map(
        att.set_index('sample_id')['n_accepted']).fillna(0).astype(int)
    cs['retained_for_C4C6C8R'] = cs['in_support']
    cs = cs.rename(columns={'path_length_edges': 'L'})
    cs[['sample_id', 'in_support', 'n_paths', 'L', 'seed_roi', 'endpoint_roi',
        'retained_for_C4C6C8R']].to_csv(
        os.path.join(P.SUP, 'COMMON_SUPPORT.csv'), index=False)

    reasons = []
    for _, r in att[~att['in_support']].iterrows():
        if r['n_candidate'] == 0:
            reason = 'ZERO_PATHS'
        elif r['L'] == 1:
            reason = 'INSUFFICIENT_PATHS_L1'
        else:
            reason = 'INSUFFICIENT_PATHS_L2'
        reasons.append({'sample_id': r['sample_id'], 'reason': reason,
                        'n_candidate': int(r['n_candidate']),
                        'n_accepted': int(r['n_accepted']),
                        'L': int(r['L']), 'seed_roi': r['seed_roi'],
                        'endpoint_roi': r['endpoint_roi']})
    pd.DataFrame(reasons).to_csv(os.path.join(P.SUP, 'ATTRITION_REASONS.csv'),
                                 index=False)

    byL = []
    for L_, g in cs.groupby('L'):
        byL.append({'L': int(L_), 'n_cells': int(len(g)),
                    'n_in_support': int(g['in_support'].sum()),
                    'frac': float(g['in_support'].mean())})
    byL_df = pd.DataFrame(byL)
    byL_df.to_csv(os.path.join(P.SUP, 'RETENTION_BY_L.csv'), index=False)

    by_seed = []
    for s_, g in cs.groupby('seed_roi'):
        by_seed.append({'seed_roi': s_, 'n_cells': int(len(g)),
                        'n_in_support': int(g['in_support'].sum()),
                        'frac': float(g['in_support'].mean())})
    by_seed_df = pd.DataFrame(by_seed)
    by_seed_df.to_csv(os.path.join(P.SUP, 'RETENTION_BY_SEED.csv'), index=False)

    fold_fracs = {}
    for (rep, fold), g in fm.groupby(['repeat', 'test_fold']):
        n_fold = len(g)
        n_sup_fold = int(g['sample_id'].isin(support_sids).sum())
        fold_fracs[f'r{rep}f{fold}'] = n_sup_fold / n_fold

    n_support = len(support_sids)
    n_seeds = uni['seed_roi'].nunique()
    seeds_retained = uni[uni['sample_id'].isin(support_sids)][
        'seed_roi'].nunique()
    audit = {
        'screen_id': P.SCREEN_ID,
        'universe_cells': 2636,
        'in_support_cells': n_support,
        'retention_frac': n_support / 2636,
        'censor': {
            'by_L': byL_df.to_dict(orient='records'),
            'by_seed': {
                'n_seed_groups_total': int(n_seeds),
                'n_seed_groups_retained': int(seeds_retained),
                'seed_frac_retained': seeds_retained / n_seeds,
                'seed_level_table': by_seed_df.to_dict(orient='records'),
            },
        },
        'fold_coverage_of_support': {
            'per_repeat_fold': {k: round(v, 4)
                                for k, v in sorted(fold_fracs.items())},
            'min_frac': round(min(fold_fracs.values()), 4),
            'mean_frac': round(float(np.mean(list(fold_fracs.values()))), 4),
        },
        'C4C6C8R_intersection': {
            'design': ('C4/C6/C8R share ONE feature matrix row set (the '
                       'in-support cells); identical sample intersection by '
                       'construction.'),
            'identical_sample_intersection': True,
        },
        'C4_retrained_on_screen_support': True,
    }
    with open(os.path.join(P.SUP, 'SUPPORT_INTERSECTION_AUDIT.json'), 'w') as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print(f'[{P.SCREEN_ID}] support: {n_support}/2636 ({n_support/2636:.1%})')
    print(f'  seeds retained: {seeds_retained}/{n_seeds} '
          f'({seeds_retained/n_seeds:.1%})')
    print(f'  fold support fracs: min '
          f'{audit["fold_coverage_of_support"]["min_frac"]}, mean '
          f'{audit["fold_coverage_of_support"]["mean_frac"]}')


if __name__ == '__main__':
    main()
