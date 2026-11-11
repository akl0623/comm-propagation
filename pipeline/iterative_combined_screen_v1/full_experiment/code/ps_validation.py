#!/usr/bin/env python3
import os
import sys
import json
import time
import hashlib
import warnings

import numpy as np
import pandas as pd

ROOT = os.environ.get('PIPELINE_ROOT', '/mnt/d/revision_v2')
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.join(PACKAGE_ROOT, 'pipeline')
sys.path[:0] = [
    os.path.join(CODE_ROOT, 'iterative_combined_screen_v1',
                 'full_experiment', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1',
                 'confirmatory', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1',
                 'candidate_screens', 'code'),
]

import stage4a_lib as A4
import ps_lib as P
from run_screens import make_matrices

warnings.filterwarnings('ignore')

N_NULL = 10000
POOL = 512
F2_CIRCUITS = ['language_arcuate', 'motor_cortico_cortical',
               'thalamocortical', 'visual_dorsal_ventral']


def parse_nodes(s):
    return [int(x) for x in s.split('|')]


def main():
    t0 = time.time()
    print('=' * 70)
    print(f'{P.SCREEN_ID} 3.4 external anatomical enrichment '
          '(frozen whitelist)')
    print('=' * 70, flush=True)
    wl = pd.read_csv(os.path.join(ROOT, 'screen_match_confirm_v1', 'config',
                                  'ANATOMICAL_WHITELIST.csv'))
    circuits = {}
    for _, r in wl.iterrows():
        if r['circuit_name'] not in F2_CIRCUITS:
            continue
        circuits.setdefault(r['circuit_name'], set()).add(int(r['roi_id']))
    circuit_names = sorted(circuits)
    assert set(circuit_names) == set(F2_CIRCUITS), (
        f'F2 circuit mismatch: {circuit_names}')
    print(f'  circuits: {circuit_names}', flush=True)

    ctx = P.load_taskC_stage4a()
    eng = ctx['eng']
    matrices = make_matrices(eng, A4.RAW, A4.BI,
                             os.path.join(ROOT, 'screen_match_confirm_v1',
                                          'config'))
    tl = matrices.tract_length
    adj = eng.mask
    reg = pd.read_parquet(P.REGISTRY)
    uni_sup = ctx['uni_sup']

    def circuit_edge_count(path_nodes, members):
        return sum(1 for i in range(1, len(path_nodes))
                   if path_nodes[i - 1] in members
                   and path_nodes[i] in members)

    real_overlap = {c: 0.0 for c in circuit_names}
    n_real_total = 0
    for _, r_ in reg.iterrows():
        nodes = parse_nodes(r_['roi_id_sequence'])
        for c in circuit_names:
            real_overlap[c] += circuit_edge_count(nodes, circuits[c])
        n_real_total += 1
    for c in circuit_names:
        real_overlap[c] /= n_real_total
    print(f'  registry paths: {n_real_total}', flush=True)

    rng0 = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f'v1{P.SCREEN_ID}validationpool'.encode()).digest()[:8], 'big'))
    pool_counts = {c: [] for c in circuit_names}
    n_cell_pool = []
    cell_ids = []
    t1 = time.time()
    for _, row in uni_sup.iterrows():
        sid = row['sample_id']
        si = eng.r2i[row['seed_roi']]
        L_ = int(row['path_length_edges'])
        gl = reg[reg['sample_id'] == sid]
        lens = []
        for s in gl['roi_id_sequence']:
            nodes = parse_nodes(s)
            lens.append(sum(tl[nodes[i - 1], nodes[i]]
                            for i in range(1, len(nodes))))
        med = float(np.median(lens)) if lens else 0.0
        rng = np.random.default_rng(rng0.integers(0, 2 ** 63))
        hemi = np.arange(0, 180) if si < 180 else np.arange(180, 360)
        lo, hi = 0.75 * med, 1.25 * med
        pool = []
        attempts = 0
        max_attempts = max(30000, 300 * POOL)
        while len(pool) < POOL and attempts < max_attempts:
            attempts += 1
            cur = si
            path = [si]
            for _ in range(L_):
                cands = [int(n) for n in hemi if adj[cur, n] == 1
                         and n not in path]
                if not cands:
                    break
                cur = int(cands[rng.integers(0, len(cands))])
                path.append(cur)
            if len(path) != L_ + 1:
                continue
            tot = sum(tl[path[i - 1], path[i]] for i in range(1, len(path)))
            if lo <= tot <= hi:
                pool.append(path)
        cell_ids.append(sid)
        n_cell_pool.append(len(pool))
        for c in circuit_names:
            pool_counts[c].append(
                np.array([circuit_edge_count(p, circuits[c]) for p in pool],
                         dtype=np.float64))
    print(f'  null pools built ({time.time()-t1:.0f}s); '
          f'min pool size {min(n_cell_pool)}', flush=True)

    draw_rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        f'v1{P.SCREEN_ID}validationnulls'.encode()).digest()[:8], 'big'))
    n_cells = len(cell_ids)
    n_real_per_cell = [len(reg[reg['sample_id'] == sid]) for sid in cell_ids]
    null_stats = {c: np.zeros(N_NULL) for c in circuit_names}
    for ci in range(n_cells):
        psz = n_cell_pool[ci]
        if psz == 0:
            continue
        n_draw = n_real_per_cell[ci]
        idx = draw_rng.integers(0, psz, size=(N_NULL, n_draw))
        for c in circuit_names:
            null_stats[c] += pool_counts[c][ci][idx].sum(axis=1)
    total_drawn = float(sum(n_real_per_cell))
    for c in circuit_names:
        null_stats[c] /= total_drawn
    print('  10,000 null path sets evaluated', flush=True)

    rows = []
    summary = {}
    boot_rng = np.random.default_rng(P.BOOT_SEED)
    for c in circuit_names:
        null_mean = float(null_stats[c].mean())
        ratio = real_overlap[c] / null_mean if null_mean > 0 else np.inf
        per_cell_real = [np.mean(
            [circuit_edge_count(parse_nodes(s), circuits[c])
             for s in reg[reg['sample_id'] == sid]['roi_id_sequence']])
            for sid in cell_ids]
        per_cell_real = np.array(per_cell_real)
        per_cell_n = np.array(n_real_per_cell, dtype=np.float64)
        B = 10000
        draw = boot_rng.integers(0, n_cells, size=(B, n_cells))
        sm = np.zeros((B, n_cells))
        for b in range(B):
            np.add.at(sm[b], draw[b], 1)
        boot_overlap = (sm * per_cell_real[None, :]).sum(axis=1) / \
            (sm * per_cell_n[None, :]).sum(axis=1)
        boot_ratio = boot_overlap / null_mean
        ci = np.percentile(boot_ratio, [2.5, 97.5])
        perm_p = (1 + int((null_stats[c] >= real_overlap[c]).sum())) / \
            (N_NULL + 1)
        rows.append({'circuit_name': c, 'real_overlap': real_overlap[c],
                     'null_mean_overlap': null_mean,
                     'enrichment_ratio': ratio, 'ci_lo': float(ci[0]),
                     'ci_hi': float(ci[1]), 'perm_p': perm_p})
        summary[c] = {'enrichment_ratio': ratio, 'perm_p': perm_p,
                      'plausible': bool(ratio > 1.0 and perm_p < 0.05)}
    df = pd.DataFrame(rows)
    assert len(df) == 4, f'expected four F2 circuits, got {len(df)}'
    order = sorted(range(len(df)), key=lambda i: df['perm_p'].iloc[i])
    adj = [float(p) for p in df['perm_p']]
    m = len(order)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, df['perm_p'].iloc[i] * (m - rank))
    for rank in range(1, m):
        adj[order[rank]] = max(adj[order[rank]], adj[order[rank - 1]])
    df['holm_perm_p'] = adj
    df['holm_family'] = df['circuit_name'].isin(F2_CIRCUITS)
    df.to_csv(os.path.join(P.VAL, 'EXTERNAL_ENRICHMENT.csv'), index=False)
    with open(os.path.join(P.VAL, 'ENRICHMENT_SUMMARY.json'), 'w') as f:
        json.dump({
            'screen_id': P.SCREEN_ID,
            'n_registry_paths': n_real_total,
            'n_null_path_sets': N_NULL,
            'pool_size_per_cell': POOL,
            'note': 'plausibility evidence only — not gold-standard '
                    'validation; frozen Stage-0 whitelist, never modified',
            'circuits': summary,
            'wall_s': round(time.time() - t0, 1),
        }, f, indent=2, sort_keys=True)
    print(df.to_string(index=False), flush=True)
    print(f'DONE ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
