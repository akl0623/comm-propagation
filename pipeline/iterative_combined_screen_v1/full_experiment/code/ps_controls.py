#!/usr/bin/env python3
import os
import sys
import json
import time
import copy
import hashlib
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
sys.path.append(os.path.join(ROOT, 'screen_match_confirm_v1',
                                'confirmatory', 'code'))
sys.path.append(os.path.join(ROOT, 'screen_match_confirm_v1',
                                'candidate_screens', 'code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stage4a_lib as A4
import ps_lib as P
import ps_screens as PS
from run_screens import make_matrices, gen_candidates
from screens import ddvs_downsample
from bridge_noscreen import (blind_from_paths, ob_from_paths, METRICS,
                             P1Engine)

warnings.filterwarnings('ignore')

N_WORKERS = 16
KSEL = 128
JAC = 0.80
G = {}

# observed real deltas read from the Stage-4A contrasts at runtime
OBS6, OBS8 = None, None


def features_from_path_lists(path_lists_by_cell, uni_sup):
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


def base_c4_features(ctx):
    uni_sup = ctx['uni_sup']
    eng = ctx['eng']
    onehot = np.zeros((len(uni_sup), 20))
    Larr = uni_sup['path_length_edges'].values.astype(np.int64)
    ok20 = Larr <= 20
    onehot[ok20] = np.eye(20)[Larr[ok20] - 1]
    seed_idx = np.array([eng.r2i[s] for s in uni_sup['seed_roi']])
    end_idx = np.array([eng.r2i[e] for e in uni_sup['endpoint_roi']])
    ep = np.stack([eng.mats[mm][seed_idx, end_idx] for mm in METRICS], axis=1)
    return np.hstack([onehot, ep])


def build_X_new(ctx, blind, ob):
    c4 = base_c4_features(ctx)
    return {'C4': c4, 'C6': np.hstack([c4, blind]),
            'C8R': np.hstack([c4, ob])}


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
    tasks = [(m, rep, fold, X_new[m]) for m in ('C6', 'C8R')
             for rep in range(5) for fold in range(5)]
    out = {'C6': [], 'C8R': []}
    with mp.Pool(N_WORKERS) as pool:
        for model, _rep, _fold, oof_rows in pool.imap_unordered(
                cv_one_fold_task, tasks):
            out[model].extend(oof_rows)
    return out


def control_bootstrap_delta(ctl_oof, model):
    oof_tc = pd.read_parquet(os.path.join(P.TC, 'OOF_PREDICTIONS.parquet'))
    c4 = oof_tc[['sample_id', 'repeat', 'outer_fold', 'y_true',
                 'c4_pred']].copy()
    c4 = c4.rename(columns={'c4_pred': 'y_pred'})
    c4['model'] = 'C4'
    cc = pd.DataFrame(ctl_oof)
    cc['model'] = model
    long = pd.concat([c4, cc], ignore_index=True)
    from ps_boot import seed_cluster_bootstrap_paired
    seed_map = P.cluster_map()
    boot = seed_cluster_bootstrap_paired(long, ['C4', model], seed_map,
                                         10000, P.BOOT_SEED)
    delta = boot[model] - boot['C4']
    n_eff = int(np.isfinite(delta).sum())
    point = _repeat_r2(cc) - _repeat_r2(c4)
    ci = np.percentile(delta[np.isfinite(delta)], [2.5, 97.5])
    return point, float(np.median(delta[np.isfinite(delta)])), float(ci[0]), \
        float(ci[1]), n_eff


def _repeat_r2(df):
    vals = [A4.metric_block(g['y_true'].values, g['y_pred'].values)['r2']
            for _, g in df.groupby('repeat')]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals))


def _valid_path(eng, cand):
    for i in range(1, len(cand)):
        u, v = cand[i - 1], cand[i]
        if eng.mask[u, v] != 1:
            return False
        for m in METRICS:
            if not np.isfinite(eng.mats[m][u, v]):
                return False
    return True


# ── control 1: order shuffle (5 repeats x per-(r,f) realizations) ──
def control_order_shuffle(eng, uni_sup, reg, ctx):
    print('[1/4] order shuffle (5 repeats x 5 folds)...', flush=True)
    path_lists_rf = {}
    for r in range(5):
        for f in range(5):
            pl = {}
            for sid, g in reg.groupby('sample_id'):
                out = []
                for s, pid in zip(g['roi_id_sequence'], g['path_id']):
                    nodes = [int(x) for x in s.split('|')]
                    rng = np.random.default_rng(P.shuffle_seed(r, f, int(pid)))
                    for attempt in range(20):
                        mids = nodes[1:-1]
                        rng.shuffle(mids)
                        cand = [nodes[0]] + mids + [nodes[-1]]
                        if _valid_path(eng, cand):
                            out.append(cand)
                            break
                    else:
                        out.append(nodes)
                pl[sid] = out
            path_lists_rf[(r, f)] = pl
    X_rf = {}
    for r in range(5):
        for f in range(5):
            blind, ob = features_from_path_lists(path_lists_rf[(r, f)],
                                                 uni_sup)
            X_rf[(r, f)] = build_X_new(ctx, blind, ob)
    G['X_rf'] = X_rf
    return path_lists_rf, X_rf


def shuffle_fold_task(args):
    model, r, f = args
    ctx = G['ctx']
    Xm = G['X_rf'][(r, f)][model]
    y = ctx['y']
    sid_pos = ctx['sid_pos']
    fm_sup = ctx['fm_sup']
    rd = fm_sup[fm_sup['repeat'] == r]
    test_sids = rd.loc[rd['test_fold'] == f, 'sample_id'].tolist()
    train_sids = rd.loc[rd['test_fold'] != f, 'sample_id'].tolist()
    tr = np.array([sid_pos[s] for s in train_sids], dtype=np.int64)
    te = np.array([sid_pos[s] for s in test_sids], dtype=np.int64)
    res = A4.train_outer_fold(Xm[tr], y[tr],
                              rd.loc[rd['test_fold'] != f,
                                     'seed_roi'].values, Xm[te])
    yt = y[te]
    oof_rows = [{'sample_id': sid, 'repeat': r, 'outer_fold': f,
                 'y_true': float(yt[i]), 'y_pred': float(res['y_pred'][i]),
                 'model': model}
                for i, sid in enumerate(test_sids)]
    mb = A4.metric_block(yt, res['y_pred'])
    return model, r, f, oof_rows, mb['r2']


def run_order_shuffle(ctx):
    ck_path = os.path.join(P.NC, '_ctl_ckpt_shuffle_oof.csv')
    det_path = os.path.join(P.NC, '_ctl_ckpt_shuffle_detailed.csv')
    if os.path.exists(ck_path) and os.path.exists(det_path):
        print('    shuffle checkpoints reused', flush=True)
        return pd.read_csv(ck_path), pd.read_csv(det_path)
    tasks = [(m, r, f) for m in ('C4', 'C6', 'C8R')
             for r in range(5) for f in range(5)]
    oof_rows = []
    det_rows = {}
    with mp.Pool(N_WORKERS) as pool:
        for model, r, f, rows, r2 in pool.imap_unordered(shuffle_fold_task,
                                                         tasks):
            oof_rows.extend(rows)
            det_rows[(model, r, f)] = r2
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(ck_path, index=False)
    det_list = []
    for r in range(5):
        for f in range(5):
            det_list.append({
                'repeat_id': r, 'fold_id': f,
                'c4_r2': det_rows[('C4', r, f)],
                'c6_r2': det_rows[('C6', r, f)],
                'c8r_r2': det_rows[('C8R', r, f)],
                'delta_c6': det_rows[('C6', r, f)] - det_rows[('C4', r, f)],
                'delta_c8r': det_rows[('C8R', r, f)] - det_rows[('C4', r, f)]})
    det = pd.DataFrame(det_list)
    det.to_csv(det_path, index=False)
    return oof, det


def _copy_matrices_with(matrices, onset=None, sc_mask=None):
    m = copy.copy(matrices)
    if onset is not None:
        m.onset = onset
    if sc_mask is not None:
        m.sc_mask = sc_mask
    return m


def screen_rerun(eng2, m2, uni_sup):
    """Re-search paths on the manipulated engine and re-run the screen with
    the frozen constants; returns per-cell post-DDVS path lists."""
    path_lists = {}
    for _, row in uni_sup.iterrows():
        sid = row['sample_id']
        si = eng2.r2i[row['seed_roi']]
        ei = eng2.r2i[row['endpoint_roi']]
        L_ = int(row['path_length_edges'])
        paths = gen_candidates(eng2, si, ei, L_, P.SCREEN_ID, sid, 512)
        accepted = PS.screen_accept(P.SCREEN_ID, paths, m2)
        accepted_post = ddvs_downsample(accepted, KSEL, JAC)
        path_lists[sid] = [list(p.nodes) for p in accepted_post]
    return path_lists


# ── control 2: latency permutation ──
def control_latency_perm(eng, matrices, uni_sup):
    print('[2/4] latency permutation...', flush=True)
    lat_perm = eng.lat.copy()
    rng = np.random.default_rng(P.ctl_seed('latperm'))
    for si in sorted(set(eng.r2i[r] for r in uni_sup['seed_roi'])):
        hemi = np.arange(0, 180) if si < 180 else np.arange(180, 360)
        lat_perm[si, hemi] = rng.permutation(lat_perm[si, hemi])
    eng2 = P1Engine()
    eng2.load()
    eng2.lat = lat_perm
    m2 = _copy_matrices_with(matrices, onset=lat_perm)
    return screen_rerun(eng2, m2, uni_sup)


# ── control 3: degree-preserving SC rewiring ──
def control_sc_rewiring(eng, matrices, uni_sup):
    print('[3/4] SC rewiring...', flush=True)
    mask = eng.mask.copy()
    rng = np.random.default_rng(P.ctl_seed('screwiring'))
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
        print(f'    hemisphere {hemi_lo}-{hemi_hi}: {swaps} swaps',
              flush=True)
    eng2 = P1Engine()
    eng2.load()
    eng2.mask = mask
    m2 = _copy_matrices_with(matrices, sc_mask=mask)
    return screen_rerun(eng2, m2, uni_sup)


# ── control 4: matched random SC walks ──
def control_matched_random(eng, matrices, uni_sup, reg):
    print('[4/4] matched random paths...', flush=True)
    tl = matrices.tract_length
    real_len = {}
    real_n = {}
    for sid, g in reg.groupby('sample_id'):
        lens = []
        for s in g['roi_id_sequence']:
            nodes = [int(x) for x in s.split('|')]
            lens.append(sum(tl[nodes[i - 1], nodes[i]]
                            for i in range(1, len(nodes))))
        real_len[sid] = float(np.median(lens)) if lens else 0.0
        real_n[sid] = len(g)
    adj = eng.mask
    path_lists = {}
    for _, row in uni_sup.iterrows():
        sid = row['sample_id']
        si = eng.r2i[row['seed_roi']]
        L_ = int(row['path_length_edges'])
        rng = np.random.default_rng(P.ctl_seed('matchedrandom', sid))
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


def main():
    global OBS6, OBS8
    t0 = time.time()
    print('=' * 70)
    print(f'{P.SCREEN_ID} 3.1/3.2 negative controls (four types)')
    print('=' * 70, flush=True)
    ctx = P.load_taskC_stage4a()
    G['ctx'] = ctx
    eng = ctx['eng']
    matrices = make_matrices(eng, A4.RAW, A4.BI,
                             os.path.join(ROOT, 'screen_match_confirm_v1',
                                          'config'))
    K0 = ctx['K0']
    uni_sup = ctx['uni_sup']
    reg = pd.read_parquet(P.REGISTRY)
    OBS6, OBS8 = P.observed_contrasts()
    print(f'  observed real deltas: C6 {OBS6:+.4f}, C8R {OBS8:+.4f}',
          flush=True)

    results = []
    for ctl_name, pl_builder in (
            ('order_shuffle', lambda: control_order_shuffle(
                eng, uni_sup, reg, ctx)),
            ('latency_permutation', lambda: control_latency_perm(
                eng, matrices, uni_sup)),
            ('sc_rewiring', lambda: control_sc_rewiring(
                eng, matrices, uni_sup)),
            ('matched_random', lambda: control_matched_random(
                eng, matrices, uni_sup, reg))):
        t1 = time.time()
        if ctl_name == 'order_shuffle':
            path_lists = pl_builder()[1]
            oof_sh, det_sh = run_order_shuffle(ctx)
            det_sh.to_csv(os.path.join(P.NC, 'ORDER_SHUFFLE_DETAILED.csv'),
                          index=False)
            for model in ('C6', 'C8R'):
                ctl_oof = oof_sh[oof_sh['model'] == model][
                    ['sample_id', 'repeat', 'outer_fold', 'y_true',
                     'y_pred']].copy()
                point, med, lo, hi, n_eff = control_bootstrap_delta(
                    ctl_oof.to_dict('records'), model)
                obs = OBS6 if model == 'C6' else OBS8
                results.append({
                    'control_type': 'order_shuffle', 'model': model,
                    'delta_r2': point, 'ci_lo': lo, 'ci_hi': hi,
                    'n_repeats': 5, 'n_effective': n_eff,
                    'replicates_real_increment': bool(point >= obs)})
        else:
            path_lists = pl_builder()
            n_paths = sum(len(v) for v in path_lists.values())
            blind, ob = features_from_path_lists(path_lists, uni_sup)
            X_new = build_X_new(ctx, blind, ob)
            ctl_oof = cv_models(X_new)
            for model in ('C6', 'C8R'):
                point, med, lo, hi, n_eff = control_bootstrap_delta(
                    ctl_oof[model], model)
                obs = OBS6 if model == 'C6' else OBS8
                results.append({
                    'control_type': ctl_name, 'model': model,
                    'delta_r2': point, 'ci_lo': lo, 'ci_hi': hi,
                    'n_effective': n_eff,
                    'replicates_real_increment': bool(point >= obs)})
        print(f'  {ctl_name} done ({time.time()-t1:.0f}s)', flush=True)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(P.NC, 'CONTROL_RESULTS.csv'), index=False)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(P.NC, 'CONTROL_RESULTS.csv'), index=False)
    key = df[df['control_type'].isin(['sc_rewiring', 'latency_permutation'])
             & (df['model'] == 'C8R')]
    hard_fail = bool((key['replicates_real_increment']).any())
    interp = df[df['control_type'].isin(['order_shuffle', 'matched_random'])
                & (df['model'] == 'C8R')]
    with open(os.path.join(P.NC, 'CONTROL_SUMMARY.json'), 'w') as f:
        json.dump({'observed': {'C6': OBS6, 'C8R': OBS8},
                   'controls': results,
                   'hard_control_replication_flag': hard_fail,
                   'note': 'sc_rewiring/latency_permutation replicating the '
                           'real C8R increment fails the primary GO gate; '
                           'order_shuffle/matched_random are interpretive '
                           'only.',
                   }, f, indent=2, sort_keys=True)
    print(json.dumps({'hard_control_replication_flag': hard_fail,
                      'interpretive': interp.to_dict('records')}, indent=2))
    print(f'CONTROLS DONE ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
