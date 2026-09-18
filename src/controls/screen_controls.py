#!/usr/bin/env python3
"""
stage4b_controls.py — 4.7 negative controls (four types).

Each control runs the SAME C4/C6/C8R modeling pipeline (features + inner
tuning + fits, 8-way parallel) with a manipulated path set; C4 features are
unchanged (the controls break only the PROCESS features / path space).
Delta R2 + 10,000-seed-cluster paired bootstrap CI per control.

  1. order_shuffle   — intermediates of each real screened path are shuffled
                      (SC-edge validity + metric finiteness enforced; the
                      temporal check is deliberately not applied because the
                      control is DEFINED to break temporal order)
  2. latency_permutation — onset latencies permuted within the seed's
                      hemisphere; full path search + B5 screen re-run
  3. sc_rewiring     — degree-preserving double-edge swaps within hemisphere;
                      full path search + B5 screen re-run
  4. matched_random  — random SC walks (latency ignored, no repeats) matched
                      on L (same cell), hemisphere, seed/endpoint degree
                      (same cell) and total tract length (+-25% of the real
                      per-cell median); features computed directly (B5
                      temporal validity is undefined for these walks by
                      construction)
"""
import os
import sys
import json
import time
import hashlib
import multiprocessing as mp
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

SMCR = os.path.join(PIPELINE_ROOT, 'screen_match_confirm_v1')
sys.path.insert(0, os.path.join(SMCR, 'confirmatory', 'code'))
sys.path.insert(0, os.path.join(SMCR, 'candidate_screens', 'code'))
sys.path.insert(0, os.path.join(SMCR, 'confirmatory', 'code'))

import stage4a_lib as A4
import stage4b_lib as L
from stage4b_lib import NC, TC, metric_block
from stage4b_boot import seed_cluster_bootstrap_paired, cluster_map
from run_screens import make_matrices
from screens import screen_B5
from bridge_noscreen import (blind_from_paths, ob_from_paths, METRICS,
                             P1Engine)

warnings.filterwarnings('ignore')

N_WORKERS = 8
G = {}


def ctl_seed(tag, sid=None):
    return int.from_bytes(hashlib.sha256(
        f'v1B5{tag}{sid or ""}'.encode()).digest()[:8], 'big')


# ──────────────────────────────────────────────────────────────────
def features_from_path_lists(eng, path_lists_by_cell, uni_sup):
    """C6/C8R sample-level features from arbitrary per-cell path lists."""
    blind_cols = [f'blind_{j:02d}' for j in range(56)]
    ob_cols = [f'ob_{m}_p{q}_{a}' for m in METRICS for q in range(4)
               for a in ('med', 'iqr')]
    blind = np.empty((len(uni_sup), 56), dtype=np.float64)
    ob = np.empty((len(uni_sup), 32), dtype=np.float64)
    for i, (_, row) in enumerate(uni_sup.iterrows()):
        sid = row['sample_id']
        L_ = int(row['path_length_edges'])
        pl = path_lists_by_cell.get(sid, [])
        if not pl:
            blind[i] = np.nan
            ob[i] = np.nan
            continue
        paths = np.asarray(pl, dtype=np.int16)
        blind[i] = blind_from_paths(paths, L_)
        od = ob_from_paths(paths, L_)
        ob[i] = [od[c] for c in ob_cols]
    return blind, ob


def cv_one_fold_task(args):
    model, rep, fold, Xm = args
    ctx = G['ctx']
    y = ctx['y']
    sid_pos = ctx['sid_pos']
    fm_sup = ctx['fm_sup']
    rd = fm_sup[fm_sup['repeat'] == rep]
    test_sids = rd.loc[rd['test_fold'] == fold, 'sample_id'].tolist()
    train_sids = rd.loc[rd['test_fold'] != fold, 'sample_id'].tolist()
    tr = np.array([sid_pos[s] for s in train_sids], dtype=np.int64)
    te = np.array([sid_pos[s] for s in test_sids], dtype=np.int64)
    res = A4.train_outer_fold(Xm[tr], y[tr],
                              rd.loc[rd['test_fold'] != fold,
                                     'seed_roi'].values, Xm[te])
    oof_rows = [{'sample_id': sid, 'repeat': rep, 'outer_fold': fold,
                 'y_true': float(y[te[i]]), 'y_pred': float(res['y_pred'][i])}
                for i, sid in enumerate(test_sids)]
    return model, rep, fold, oof_rows


def cv_models(X_new):
    """Full inner-tuned CV for the two process models (per-fold parallel)."""
    tasks = [(m, rep, fold, X_new[m]) for m in ('C6', 'C8R')
             for rep in range(5) for fold in range(5)]
    out = {'C6': [], 'C8R': []}
    with mp.Pool(N_WORKERS) as pool:
        for model, _rep, _fold, oof_rows in pool.imap_unordered(
                cv_one_fold_task, tasks):
            out[model].extend(oof_rows)
    return out


def control_bootstrap_delta(ctl_oof, model):
    """Paired cluster bootstrap delta of a control model vs the real C4."""
    oof_tc = pd.read_parquet(os.path.join(TC, 'OOF_PREDICTIONS.parquet'))
    c4 = oof_tc[['sample_id', 'repeat', 'outer_fold', 'y_true', 'c4_pred']].copy()
    c4 = c4.rename(columns={'c4_pred': 'y_pred'})
    c4['model'] = 'C4'
    cc = pd.DataFrame(ctl_oof)
    cc['model'] = model
    long = pd.concat([c4, cc], ignore_index=True)
    boot = seed_cluster_bootstrap_paired(long, ['C4', model], 5,
                                         cluster_map(), 10000, L.BOOT_SEED)
    delta = boot[model] - boot['C4']
    n_eff = int(np.isfinite(delta).sum())
    point = _repeat_r2(cc) - _repeat_r2(c4)
    ci = np.percentile(delta[np.isfinite(delta)], [2.5, 97.5])
    return point, float(np.median(delta[np.isfinite(delta)])), float(ci[0]), \
        float(ci[1]), n_eff


def _repeat_r2(df):
    vals = [metric_block(g['y_true'].values, g['y_pred'].values)['r2']
            for _, g in df.groupby('repeat')]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals))


# ──────────────────────────────────────────────────────────────────
# control 1: order shuffle
# ──────────────────────────────────────────────────────────────────
def control_order_shuffle(eng, matrices, uni_sup, reg):
    print('[1/4] order shuffle...', flush=True)
    path_lists = {}
    for sid, g in reg.groupby('sample_id'):
        rng = np.random.default_rng(ctl_seed('ordershuffle', sid))
        pl = []
        for s in g['roi_id_sequence']:
            nodes = [int(x) for x in s.split('|')]
            L_ = len(nodes) - 1
            for attempt in range(20):
                mids = nodes[1:-1]
                rng.shuffle(mids)
                cand = [nodes[0]] + mids + [nodes[-1]]
                ok = True
                for i in range(1, len(cand)):
                    u, v = cand[i - 1], cand[i]
                    if matrices.sc_mask[u, v] != 1:
                        ok = False
                        break
                    for m in METRICS:
                        if not np.isfinite(getattr(matrices, m)[u, v]):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    pl.append(cand)
                    break
            else:
                pl.append(nodes)  # keep original if no valid shuffle found
        path_lists[sid] = pl
    return path_lists


# ──────────────────────────────────────────────────────────────────
# control 2: latency permutation
# ──────────────────────────────────────────────────────────────────
def control_latency_perm(eng, matrices, uni_sup, b5p, K0):
    print('[2/4] latency permutation...', flush=True)
    lat_perm = eng.lat.copy()
    rng = np.random.default_rng(ctl_seed('latperm'))
    for si in sorted(set(eng.r2i[r] for r in uni_sup['seed_roi'])):
        hemi = np.arange(0, 180) if si < 180 else np.arange(180, 360)
        lat_perm[si, hemi] = rng.permutation(lat_perm[si, hemi])
    eng2 = P1Engine()
    eng2.load()
    eng2.lat = lat_perm
    m2 = _copy_matrices_with(matrices, onset=lat_perm)
    path_lists = {}
    for _, row in uni_sup.iterrows():
        sid = row['sample_id']
        si = eng2.r2i[row['seed_roi']]
        ei = eng2.r2i[row['endpoint_roi']]
        L_ = int(row['path_length_edges'])
        paths = A4.gen_candidates(eng2, si, ei, L_, 'B5', sid, K0)
        accepted = screen_B5(paths, b5p, m2)
        path_lists[sid] = [list(p.nodes) for p in accepted]
    return path_lists


# ──────────────────────────────────────────────────────────────────
# control 3: degree-preserving SC rewiring
# ──────────────────────────────────────────────────────────────────
def control_sc_rewiring(eng, matrices, uni_sup, b5p, K0):
    print('[3/4] SC rewiring...', flush=True)
    mask = eng.mask.copy()
    rng = np.random.default_rng(ctl_seed('screwiring'))
    for hemi_lo, hemi_hi in ((0, 180), (180, 360)):
        nodes = np.arange(hemi_lo, hemi_hi)
        idx = np.triu_indices(len(nodes), 1)
        pairs = [(int(nodes[i]), int(nodes[j])) for i, j in zip(*idx)
                 if mask[nodes[i], nodes[j]] == 1]
        n_swaps = 4 * len(pairs)
        swaps = 0
        attempts = 0
        while swaps < n_swaps and attempts < 20 * n_swaps and len(pairs) >= 2:
            attempts += 1
            a, b = pairs[int(rng.integers(0, len(pairs)))]
            c, d = pairs[int(rng.integers(0, len(pairs)))]
            if len({a, b, c, d}) < 4:
                continue
            if mask[a, d] == 1 or mask[c, b] == 1:
                continue
            mask[a, b] = mask[b, a] = 0
            mask[c, d] = mask[d, c] = 0
            mask[a, d] = mask[d, a] = 1
            mask[c, b] = mask[b, c] = 1
            pairs.remove((a, b))
            pairs.remove((c, d))
            pairs.append(tuple(sorted((a, d))))
            pairs.append(tuple(sorted((c, b))))
            swaps += 1
    eng2 = P1Engine()
    eng2.load()
    eng2.mask = mask
    m2 = _copy_matrices_with(matrices, sc_mask=mask)
    path_lists = {}
    for _, row in uni_sup.iterrows():
        sid = row['sample_id']
        si = eng2.r2i[row['seed_roi']]
        ei = eng2.r2i[row['endpoint_roi']]
        L_ = int(row['path_length_edges'])
        paths = A4.gen_candidates(eng2, si, ei, L_, 'B5', sid, K0)
        accepted = screen_B5(paths, b5p, m2)
        path_lists[sid] = [list(p.nodes) for p in accepted]
    return path_lists


def _copy_matrices_with(matrices, onset=None, sc_mask=None):
    """Shallow copy of the Matrices namespace with optional overrides."""
    import copy
    m = copy.copy(matrices)
    if onset is not None:
        m.onset = onset
    if sc_mask is not None:
        m.sc_mask = sc_mask
    return m


# ──────────────────────────────────────────────────────────────────
# control 4: matched random SC walks (latency ignored)
# ──────────────────────────────────────────────────────────────────
def control_matched_random(eng, matrices, uni_sup, reg):
    print('[4/4] matched random paths...', flush=True)
    # real per-cell median total tract length
    tl = matrices.tract_length
    real_len = {}
    real_n = {}
    for sid, g in reg.groupby('sample_id'):
        lens = []
        for s in g['roi_id_sequence']:
            nodes = [int(x) for x in s.split('|')]
            lens.append(sum(tl[nodes[i - 1], nodes[i]]
                            for i in range(1, len(nodes))))
        real_len[sid] = float(np.median(lens))
        real_n[sid] = len(g)
    adj = eng.mask
    path_lists = {}
    for _, row in uni_sup.iterrows():
        sid = row['sample_id']
        si = eng.r2i[row['seed_roi']]
        L_ = int(row['path_length_edges'])
        rng = np.random.default_rng(ctl_seed('matchedrandom', sid))
        hemi = np.arange(0, 180) if si < 180 else np.arange(180, 360)
        target = min(real_n.get(sid, 0), 512)
        if target == 0:
            path_lists[sid] = []
            continue
        lo, hi = 0.75 * real_len[sid], 1.25 * real_len[sid]
        out = []
        attempts = 0
        max_attempts = max(20000, 200 * target)
        while len(out) < target and attempts < max_attempts:
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
                out.append(path)
        path_lists[sid] = out
    return path_lists


# ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 70)
    print('4.7 negative controls')
    print('=' * 70, flush=True)
    ctx = L.load_taskC_stage4a()
    G['ctx'] = ctx
    eng = ctx['eng']
    matrices = make_matrices(eng, A4.RAW, A4.BI, os.path.join(SMCR, 'config'))
    b5p = A4.b5_params_from_config()
    K0 = ctx['K0']
    uni_sup = ctx['uni_sup']
    reg = pd.read_parquet(os.path.join(SMCR, 'confirmatory',
                                       'path_registry.parquet'))

    # observed real deltas
    obs6 = 0.0015410439322268
    obs8 = 0.0126973860360123

    results = []
    for ctl_name, pl_builder in (
            ('order_shuffle', lambda: control_order_shuffle(
                eng, matrices, uni_sup, reg)),
            ('latency_permutation', lambda: control_latency_perm(
                eng, matrices, uni_sup, b5p, K0)),
            ('sc_rewiring', lambda: control_sc_rewiring(
                eng, matrices, uni_sup, b5p, K0)),
            ('matched_random', lambda: control_matched_random(
                eng, matrices, uni_sup, reg))):
        t1 = time.time()
        path_lists = pl_builder()
        n_paths = sum(len(v) for v in path_lists.values())
        blind, ob = features_from_path_lists(eng, path_lists, uni_sup)
        Lb = [f'blind_{j:02d}' for j in range(56)]
        Obb = [f'ob_{m}_p{q}_{a}' for m in METRICS for q in range(4)
               for a in ('med', 'iqr')]
        n_cells_ok = int(np.isfinite(blind).all(axis=1).sum())
        onehot = np.zeros((len(uni_sup), 20))
        Larr = uni_sup['path_length_edges'].values.astype(np.int64)
        ok20 = Larr <= 20
        onehot[ok20] = np.eye(20)[Larr[ok20] - 1]
        seed_idx = np.array([eng.r2i[s] for s in uni_sup['seed_roi']])
        end_idx = np.array([eng.r2i[e] for e in uni_sup['endpoint_roi']])
        ep = np.stack([eng.mats[mm][seed_idx, end_idx] for mm in METRICS],
                      axis=1)
        X_new = {'C6': np.hstack([onehot, ep, blind]),
                 'C8R': np.hstack([onehot, ep, ob])}
        print(f'  {ctl_name}: {n_paths} paths, {n_cells_ok}/{len(uni_sup)} '
              f'cells with features ({time.time()-t1:.0f}s)', flush=True)
        ctl_oof = cv_models(X_new)
        for model in ('C6', 'C8R'):
            point, med, lo, hi, n_eff = control_bootstrap_delta(
                ctl_oof[model], model)
            obs = obs6 if model == 'C6' else obs8
            results.append({
                'control_type': ctl_name, 'model': model,
                'delta_r2': point, 'ci_lo': lo, 'ci_hi': hi,
                'n_effective': n_eff,
                'replicates_real_increment': bool(point >= obs)})
            print(f'    {model}: dR2={point:+.4f} [{lo:+.4f},{hi:+.4f}] '
                  f'replicates_real={point >= obs}', flush=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(NC, 'CONTROL_RESULTS.csv'), index=False)
    # artifact check: key controls replicating the real increment
    key = df[df['control_type'].isin(['order_shuffle', 'matched_random'])
             & (df['model'] == 'C8R')]
    artifact = bool((key['replicates_real_increment']).any())
    with open(os.path.join(NC, 'CONTROL_SUMMARY.json'), 'w') as f:
        json.dump({'observed': {'C6': obs6, 'C8R': obs8},
                   'controls': results,
                   'artifact_flag': artifact,
                   'note': 'order_shuffle/matched_random replicating the real '
                           'C8R increment would mark the result an artifact'},
                  f, indent=2, sort_keys=True)
    print(json.dumps({'artifact_flag': artifact}, indent=2))
    print(f'CONTROLS DONE ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
