#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pl_lib as PL                                        # noqa: E402

FEAT = os.path.join(PL.PLE, 'features')
os.makedirs(FEAT, exist_ok=True)

# variant -> (path-set parquet, permutation parquet or None, block)
VARIANTS = {
    'C4':                   (None, None, 'C4'),
    'C8R-All':              ('all_sampled_paths.parquet', None, 'C8R'),
    'C8R-S10Path':          ('S10_accepted_paths.parquet', None, 'C8R'),
    'C8R-OrderShuffle1':    ('order_shuffle_paths_rep1.parquet', None, 'C8R'),
    'C8R-OrderShuffle2':    ('order_shuffle_paths_rep2.parquet', None, 'C8R'),
    'C8R-OrderShuffle3':    ('order_shuffle_paths_rep3.parquet', None, 'C8R'),
    'C8R-EdgeShuffle1':     ('S10_accepted_paths.parquet',
                             'edge_order_shuffle_perms_rep1.parquet', 'C8R'),
    'C8R-EdgeShuffle2':     ('S10_accepted_paths.parquet',
                             'edge_order_shuffle_perms_rep2.parquet', 'C8R'),
    'C8R-EdgeShuffle3':     ('S10_accepted_paths.parquet',
                             'edge_order_shuffle_perms_rep3.parquet', 'C8R'),
    'C8R-MatchedRandom':    ('matched_random_paths.parquet', None, 'C8R'),
    'C8R-SCNull':           ('sc_null_paths.parquet', None, 'C8R'),
    'C8R-LatencyNull':      ('latency_null_paths.parquet', None, 'C8R'),
    'C6-All':               ('all_sampled_paths.parquet', None, 'C6'),
    'C6-S10Path':           ('S10_accepted_paths.parquet', None, 'C6'),
}

SHUFFLE_ARMS = ['C8R-OrderShuffle1', 'C8R-OrderShuffle2', 'C8R-OrderShuffle3']
EDGE_ARMS = ['C8R-EdgeShuffle1', 'C8R-EdgeShuffle2', 'C8R-EdgeShuffle3']

SMOKE_VARIANTS = (['C4', 'C8R-All', 'C8R-S10Path', 'C8R-MatchedRandom']
                  + SHUFFLE_ARMS + EDGE_ARMS)
FULL_VARIANTS = list(VARIANTS)


def load_perms(fname):
    df = pd.read_parquet(os.path.join(PL.PSETS, fname))
    out = {}
    for sid, g in df.groupby('sample_id', sort=False):
        out[sid] = [[int(x) for x in s.split('|')]
                    for s in g.sort_values('path_index')['edge_order']]
    return out


def build_variant_X(ctx, variant, c4):
    """Design matrix for one variant, cached as .npy under features/."""
    cache = os.path.join(FEAT, f'X_{variant}.npy')
    npc = os.path.join(FEAT, f'npaths_{variant}.npy')
    if os.path.exists(cache):
        return np.load(cache), np.load(npc)
    pset, pperm, block = VARIANTS[variant]
    if pset is None:                      # C4: no path information
        X, n_paths = c4, np.zeros(len(ctx['uni']), dtype=np.int64)
    else:
        path_lists = PL.load_path_set(pset)
        perm_lists = load_perms(pperm) if pperm else None
        Xd, n_paths = PL.build_X(ctx, path_lists, c4=c4, perm_lists=perm_lists)
        X = Xd[block]
    np.save(cache, X)
    np.save(npc, n_paths)
    return X, n_paths
