#!/usr/bin/env python3
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, 'cucr_secondary_analysis_v1')
sys.path.append(os.path.join(OUT, 'code'))

CM = os.path.join(OUT, 'communication_patterns')
PAR = os.path.join(CM, '04_participation')
MATS = os.path.join(PAR, 'matrices')
FIGS = os.path.join(PAR, 'figures')
os.makedirs(MATS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

PATHLIB = os.path.join(OUT, 'path_library', 'CUCR_ALL_SAMPLED_PATHS.parquet')
META_EXACT = os.path.join(OUT, 'features', 'metadata_exact.parquet')
CONTRIB = os.path.join(CM, '03_contribution',
                       'cucr_sample_level_contributions.parquet')
YEO7_P = os.path.join(ROOT, 'raw_inputs', 'mmp_to_yeo7_mapping.csv')
ROI_P = os.path.join(ROOT, 'frozen_s0', 'bridge_inputs', 'roi_order_C0R2C.txt')

LENGTH_GROUPS = {'short': (1, 5), 'medium': (6, 9), 'long': (10, 13)}
NETWORK_ORDER = ['Visual', 'Somatomotor', 'Dorsal Attention',
                 'Ventral Attention', 'Limbic', 'Frontoparietal', 'Default']
CLASSES = ['all', 'persistent_routing', 'partial_diffusion']
SOURCES = ['FULLCOEF', 'MEDIANCOEF']


def log(msg):
    print(msg, flush=True)


def main():
    with open(ROI_P) as f:
        roi = [l.strip() for l in f if l.strip()]
    yeo = pd.read_csv(YEO7_P)
    yeo['region'] = yeo['MMP_Label'] + '_ROI'
    r2net = dict(zip(yeo['region'], yeo['Yeo7_Network_Label']))
    order = {n: i for i, n in enumerate(NETWORK_ORDER)}
    sortk = lambda r: (order.get(r2net.get(r, 'Unknown'), len(order)), r)

    meta = pd.read_parquet(META_EXACT)
    support_sids = set(meta['sample_id'])
    pl = pd.read_parquet(PATHLIB)
    pl = pl[pl['sample_id'].isin(support_sids)].reset_index(drop=True)
    pl['path_nodes'] = pl['roi_id_sequence'].str.split('|').map(
        lambda s: [roi[int(x)] for x in s])
    pl['start'] = pl['path_nodes'].map(lambda s: s[0])
    pl['L'] = pl['path_nodes'].map(len) - 1
    pl['path_id'] = pl['sample_id'] + '#' + pl['path_index'].astype(str)
    log('CUCR paths: %d, cells: %d' % (len(pl), pl['sample_id'].nunique()))

    c = pd.read_parquet(CONTRIB, columns=['path_id', 'length',
                                          'dominant_type', 'coef_source',
                                          'coef_zero_count'])
    flags = {}
    for src in SOURCES:
        d = c[c['coef_source'] == src].copy()
        d = d[d['coef_zero_count'] != 4]
        g = d.groupby('path_id', sort=False)
        agg = g.agg(n_diffusion_subpaths=(
            'dominant_type', lambda s: int((s == 'diffusion').sum()))).reset_index()
        agg['is_partial_diffusion'] = agg['n_diffusion_subpaths'] > 0
        flags[src] = agg[['path_id', 'is_partial_diffusion']]
    pl = pl.merge(flags['FULLCOEF'].rename(
        columns={'is_partial_diffusion': 'part_FULL'}), on='path_id',
        how='left')
    pl = pl.merge(flags['MEDIANCOEF'].rename(
        columns={'is_partial_diffusion': 'part_MED'}), on='path_id',
        how='left')
    assert pl['part_FULL'].notna().all() and pl['part_MED'].notna().all()

    seeds_present = sorted(set(pl['start']))
    all_through = set()
    for nodes in pl['path_nodes']:
        all_through.update(nodes)
    seed_rows = sorted(seeds_present, key=sortk)
    seed_pos = {r: k for k, r in enumerate(seed_rows)}
    cols_present = sorted(all_through, key=sortk)
    col_pos = {r: k for k, r in enumerate(cols_present)}
    pl['seed_pos'] = pl['start'].map(seed_pos)

    manifest_rows = []
    for gname, (lo, hi) in LENGTH_GROUPS.items():
        g = pl[(pl['L'] >= lo) & (pl['L'] <= hi)]
        for cls in CLASSES:
            for src in SOURCES:
                col = 'part_FULL' if src == 'FULLCOEF' else 'part_MED'
                if cls == 'all':
                    sub = g
                elif cls == 'persistent_routing':
                    sub = g[~g[col]]
                else:
                    sub = g[g[col]]
                if len(sub) == 0:
                    continue
                mat = np.zeros((len(seed_rows), len(cols_present)))
                for nodes, sp_ in zip(sub['path_nodes'], sub['seed_pos']):
                    for through in nodes[1:]:
                        if through in col_pos:
                            mat[sp_, col_pos[through]] += 1
                row_sums = mat.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                norm = mat / row_sums
                pd.DataFrame(mat, index=pd.Index(seed_rows, name='seed_roi'),
                             columns=pd.Index(cols_present,
                                              name='through_roi')).to_csv(
                    os.path.join(MATS, 'through_counts_%s_%s_%s.csv'
                                 % (gname, cls, src)))
                pd.DataFrame(norm, index=pd.Index(seed_rows, name='seed_roi'),
                             columns=pd.Index(cols_present,
                                              name='through_roi')).to_csv(
                    os.path.join(MATS, 'through_proportion_%s_%s_%s.csv'
                                 % (gname, cls, src)))
                manifest_rows.append({
                    'length_group': gname, 'class': cls, 'coef_source': src,
                    'n_paths': len(sub), 'n_seeds': int(sub['start'].nunique()),
                    'L_min': int(sub['L'].min()), 'L_max': int(sub['L'].max())})
                # heatmap
                fig, ax = plt.subplots(figsize=(11, 8))
                im = ax.imshow(norm, aspect='auto', cmap='viridis', vmin=0,
                               vmax=min(0.5, float(np.percentile(norm, 99))))
                ax.set_title('CUCR through-region proportion | %s | %s | %s\n'
                             '(%d paths, %d seeds)'
                             % (gname, cls, src, len(sub),
                                sub['start'].nunique()), fontsize=9)
                ax.set_xlabel('through ROI (sorted by Yeo7 network)')
                ax.set_ylabel('seed ROI (sorted by Yeo7 network)')
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(os.path.join(FIGS, 'through_%s_%s_%s.png'
                                         % (gname, cls, src)), dpi=150)
                plt.close(fig)
                log('%s/%s/%s: %d paths, %d seeds'
                    % (gname, cls, src, len(sub), sub['start'].nunique()))
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(PAR, 'CUCR_PARTICIPATION_MANIFEST.csv'),
                    index=False)
    log('CUCR participation done: %d matrices' % len(manifest))


if __name__ == '__main__':
    main()
