#!/usr/bin/env python3
import os
import sys
import json
import copy
import hashlib
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
SMCR = os.path.join(ROOT, 'screen_match_confirm_v1')
ICS = os.path.join(ROOT, 'iterative_combined_screen_v1')
sys.path.append(os.path.join(SMCR, 'confirmatory', 'code'))
sys.path.append(os.path.join(SMCR, 'candidate_screens', 'code'))
sys.path.append(os.path.join(ICS, 'full_experiment', 'code'))

import stage4a_lib as A4                                   # noqa: E402
import bridge_noscreen as B                                # noqa: E402
from bridge_noscreen import (P1Engine, blind_from_paths,   # noqa: E402
                             ob_from_paths, METRICS, cell_seed)
import ps_screens as PS                                    # noqa: E402
from screens import Path, ddvs_downsample                  # noqa: E402
from run_screens import gen_candidates, make_matrices      # noqa: E402

warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────── output tree
OUT = os.path.join(ROOT, 's10_pathlevel_v1')
CFG = os.path.join(OUT, 'config')
AUD = os.path.join(OUT, 'audit')
PLE = os.path.join(OUT, 'path_level_experiment')
PSETS = os.path.join(PLE, 'path_sets')
SMOKE = os.path.join(PLE, 'smoke_test')
RES = os.path.join(PLE, 'results')
INF = os.path.join(PLE, 'inference')
GATES = os.path.join(OUT, 'gates')
for _d in (CFG, AUD, PSETS, SMOKE, RES, INF, GATES):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────── frozen S10 inputs
SCREEN_ID = 'S10'
S10_THRESHOLD = 0.41549280879846406
KSEL = 128
JAC = 0.80
K0_SAMPLE = 512
BOOT_SEED = 8769577528861415417        # frozen Task C cluster-bootstrap seed
N_SHUFFLE_REPS = 3

SMK = os.path.join(ICS, 'smoke_test', 'ROUND2')
REGISTRY_P = os.path.join(SMK, 'S10_registry.parquet')
ATTRITION_P = os.path.join(SMK, 'S10_attrition.csv')
PRIOR_TASKC = os.path.join(ICS, 'full_experiment', 'S10', 'taskC')

METRIC_KEYS = ('r2', 'rmse', 'mae', 'pearson_r', 'calib_intercept',
               'calib_slope')
OB_COLS = [f'ob_{m}_p{q}_{a}' for m in METRICS for q in range(4)
           for a in ('med', 'iqr')]


def log(msg):
    print(msg, flush=True)


def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def seed_from(*parts):
    """Deterministic 64-bit seed from a descriptive string."""
    key = '|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def feature_cell_seed(sid):
    """Frozen prior-run feature seed: SHA256('v1' + 'S10' + cell_id)[:8].

    Reproduces the C8R-All path sample of the prior full experiment exactly.
    """
    return int.from_bytes(
        hashlib.sha256(f'v1{SCREEN_ID}{sid}'.encode()).digest()[:8], 'big')


# ══════════════════════════════════════════════════════════════════
# universe / support / engine
# ══════════════════════════════════════════════════════════════════
def load_context(with_matrices=False):
    """Engine + support universe + fold manifest + target, in a fixed order."""
    eng = A4.build_engine()          # also sets bridge_noscreen.ENGINE
    uni_full = pd.read_csv(B.UNI_P)
    fm = pd.read_csv(B.FOLD_P)
    att = pd.read_csv(ATTRITION_P)
    support_sids = set(att[att['in_support']]['sample_id'])
    uni = uni_full[uni_full['sample_id'].isin(support_sids)].reset_index(
        drop=True)
    uni = uni[['sample_id', 'seed_roi', 'endpoint_roi', 'path_length_edges']]
    y = uni['sample_id'].map(
        dict(zip(uni_full['sample_id'],
                 uni_full['y_endpoint'].astype(np.float64)))
    ).values.astype(np.float64)
    ctx = {
        'eng': eng,
        'uni': uni,
        'uni_full': uni_full,
        'y': y,
        'sid_pos': {sid: i for i, sid in enumerate(uni['sample_id'])},
        'fm_sup': fm[fm['sample_id'].isin(support_sids)].reset_index(drop=True),
        'support_sids': support_sids,
        'attrition': att,
    }
    if with_matrices:
        ctx['matrices'] = make_matrices(eng, B.RAW, B.BI,
                                        os.path.join(SMCR, 'config'))
    return ctx


def sc_weight_matrix():
    return pd.read_csv(B.SC_P, header=None).values.astype(np.float64)


def path_mean_sc(nodes, sc):
    n = np.asarray(nodes, dtype=np.int64)
    return float(np.mean(sc[n[:-1], n[1:]]))


# ══════════════════════════════════════════════════════════════════
# path-set I/O
# ══════════════════════════════════════════════════════════════════
def path_lists_to_frame(path_lists, variant):
    rows = []
    for sid, pl in path_lists.items():
        for k, nodes in enumerate(pl):
            rows.append({'sample_id': sid, 'path_index': k,
                         'L': len(nodes) - 1,
                         'roi_id_sequence': '|'.join(str(int(x))
                                                     for x in nodes),
                         'variant': variant})
    return pd.DataFrame(rows)


def frame_to_path_lists(df):
    out = {}
    for sid, g in df.groupby('sample_id', sort=False):
        out[sid] = [[int(x) for x in s.split('|')]
                    for s in g.sort_values('path_index')['roi_id_sequence']]
    return out


def save_path_set(path_lists, variant, fname=None):
    df = path_lists_to_frame(path_lists, variant)
    p = os.path.join(PSETS, fname or f'{variant}.parquet')
    df.to_parquet(p, index=False)
    return p, df


def load_path_set(fname):
    return frame_to_path_lists(pd.read_parquet(os.path.join(PSETS, fname)))


# ══════════════════════════════════════════════════════════════════
# path-set builders
# ══════════════════════════════════════════════════════════════════
def build_s10_accepted(ctx):
    """Post-DDVS S10-accepted paths per supported cell (from the frozen
    registry — the exact set the screen selected)."""
    reg = pd.read_parquet(REGISTRY_P)
    reg = reg[reg['sample_id'].isin(ctx['support_sids'])]
    out = {}
    for sid, g in reg.groupby('sample_id', sort=False):
        g = g.sort_values('path_rank')
        out[sid] = [[int(x) for x in s.split('|')]
                    for s in g['roi_id_sequence']]
    # cells with no accepted path cannot be in support by construction
    missing = ctx['support_sids'] - set(out)
    assert not missing, f'{len(missing)} supported cells absent from registry'
    return out


def build_all_sampled(ctx):
    """K0=512 ranks sampled over the cell's FULL simple-path space with the
    frozen prior-run feature seed — reproduces C8R-All bit-for-bit."""
    eng = ctx['eng']
    out = {}
    for _, r in ctx['uni'].iterrows():
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        ranks = eng.sample_ranks(si, ei, L, K0_SAMPLE,
                                 feature_cell_seed(r['sample_id']))
        arr = eng.unrank_cell(si, ei, L, ranks)
        out[r['sample_id']] = [[int(x) for x in row] for row in arr]
    return out


def _sc_and_metrics_valid(eng, cand):
    """Validity used for order shuffling: every consecutive pair must be an SC
    edge with finite metrics.  Strict temporal ordering is deliberately NOT
    required — enforcing it would make the identity permutation the only legal
    ordering (DAG nodes are latency-sorted), collapsing the control."""
    for i in range(1, len(cand)):
        u, v = cand[i - 1], cand[i]
        if eng.mask[u, v] != 1:
            return False
        for m in METRICS:
            if not np.isfinite(eng.mats[m][u, v]):
                return False
    return True


def build_order_shuffle(ctx, s10_paths, rep):
    """Intermediate ROIs of each S10-accepted path randomly permuted; seed and
    endpoint fixed.  Falls back to the original ordering after 20 failed
    attempts (recorded)."""
    eng = ctx['eng']
    out = {}
    n_total = n_fallback = n_identity = 0
    for sid, pl in s10_paths.items():
        rows = []
        for k, nodes in enumerate(pl):
            n_total += 1
            if len(nodes) <= 3:            # L<=2: no reorderable interior
                rows.append(list(nodes))
                n_identity += 1
                continue
            rng = np.random.default_rng(
                seed_from('pl_v1_order_shuffle', SCREEN_ID, rep, sid, k))
            placed = None
            for _ in range(20):
                mids = list(nodes[1:-1])
                rng.shuffle(mids)
                cand = [nodes[0]] + mids + [nodes[-1]]
                if cand == list(nodes):
                    continue
                if _sc_and_metrics_valid(eng, cand):
                    placed = cand
                    break
            if placed is None:
                placed = list(nodes)
                n_fallback += 1
            rows.append(placed)
        out[sid] = rows
    stats = {'n_paths': n_total, 'n_fallback_to_original': n_fallback,
             'n_no_interior_L_le_2': n_identity,
             'frac_effectively_shuffled': (
                 (n_total - n_fallback - n_identity) / n_total
                 if n_total else 0.0)}
    return out, stats


def build_edge_order_shuffle(ctx, s10_paths, rep):
    """Co-primary order control: the path's EDGE SEQUENCE is permuted while the
    edge multiset — and therefore every edge's metric values — is preserved
    exactly.  Satisfies 'seed/endpoint/edge set unchanged, order randomised'
    literally, has no validity constraint and hence no fallback, so it is not
    diluted at long L the way the node permutation is.  Stored as an explicit
    per-path permutation of edge positions."""
    out = {}
    n_total = n_identity = 0
    for sid, pl in s10_paths.items():
        perms = []
        for k, nodes in enumerate(pl):
            L = len(nodes) - 1
            n_total += 1
            if L < 2:
                perms.append(list(range(L)))
                n_identity += 1
                continue
            rng = np.random.default_rng(
                seed_from('pl_v1_edge_order_shuffle', SCREEN_ID, rep, sid, k))
            p = np.arange(L)
            for _ in range(20):
                rng.shuffle(p)
                if not np.array_equal(p, np.arange(L)):
                    break
            perms.append([int(x) for x in p])
        out[sid] = perms
    stats = {'n_paths': n_total, 'n_no_interior_L_lt_2': n_identity,
             'n_fallback_to_original': 0,
             'frac_effectively_shuffled': ((n_total - n_identity) / n_total
                                           if n_total else 0.0)}
    return out, stats


def build_matched_random(ctx, s10_paths, sc):
    """Random paths from the SAME candidate pool with the variance screen
    REMOVED, DDVS applied identically, then greedily matched to the S10 set on
    per-path mean SC (count and L match by construction)."""
    eng = ctx['eng']
    out = {}
    diag = []
    for _, r in ctx['uni'].iterrows():
        sid = r['sample_id']
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        target = s10_paths[sid]
        n_t = len(target)
        pool = gen_candidates(eng, si, ei, L, SCREEN_ID, sid, K0_SAMPLE)
        pool_post = ddvs_downsample(pool, KSEL, JAC)
        pool_nodes = [list(p.nodes) for p in pool_post]
        tgt_sc = np.array([path_mean_sc(p, sc) for p in target])
        pool_sc = np.array([path_mean_sc(p, sc) for p in pool_nodes])
        # greedy nearest-neighbour matching on mean SC, without replacement
        order = np.argsort(tgt_sc, kind='stable')
        used = np.zeros(len(pool_nodes), dtype=bool)
        sel = []
        for i in order:
            if used.all():
                break
            d = np.abs(pool_sc - tgt_sc[i])
            d[used] = np.inf
            j = int(np.argmin(d))
            used[j] = True
            sel.append(pool_nodes[j])
        out[sid] = sel
        diag.append({'sample_id': sid, 'L': L, 'n_s10': n_t,
                     'n_pool_post_ddvs': len(pool_nodes),
                     'n_matched': len(sel),
                     'mean_sc_s10': float(np.mean(tgt_sc)) if n_t else np.nan,
                     'mean_sc_matched': (float(np.mean([path_mean_sc(p, sc)
                                                        for p in sel]))
                                         if sel else np.nan),
                     'overlap_with_s10': len(
                         {tuple(p) for p in sel}
                         & {tuple(p) for p in target})})
    return out, pd.DataFrame(diag)


def _copy_matrices_with(matrices, onset=None, sc_mask=None):
    m = copy.copy(matrices)
    if onset is not None:
        m.onset = onset
    if sc_mask is not None:
        m.sc_mask = sc_mask
    return m


def _screen_rerun(eng2, m2, uni):
    """Re-search paths on a manipulated engine, re-apply the FROZEN S10 rule,
    then DDVS — identical construction to the real screen."""
    out = {}
    for _, r in uni.iterrows():
        sid = r['sample_id']
        si, ei = eng2.r2i[r['seed_roi']], eng2.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        paths = gen_candidates(eng2, si, ei, L, SCREEN_ID, sid, K0_SAMPLE)
        accepted = PS.screen_accept(SCREEN_ID, paths, m2)
        out[sid] = [list(p.nodes) for p in ddvs_downsample(accepted, KSEL, JAC)]
    return out


def build_latency_null(ctx):
    """Onset permuted within the seed's hemisphere, then full re-search +
    frozen-constant S10 re-run."""
    eng = ctx['eng']
    lat_perm = eng.lat.copy()
    rng = np.random.default_rng(seed_from('pl_v1_latency_null', SCREEN_ID))
    for si in sorted({eng.r2i[r] for r in ctx['uni']['seed_roi']}):
        hemi = np.arange(0, 180) if si < 180 else np.arange(180, 360)
        lat_perm[si, hemi] = rng.permutation(lat_perm[si, hemi])
    eng2 = P1Engine()
    eng2.load()
    eng2.lat = lat_perm
    m2 = _copy_matrices_with(ctx['matrices'], onset=lat_perm)
    return _screen_rerun(eng2, m2, ctx['uni'])


def build_sc_null(ctx):
    """Degree-preserving double-edge swaps within hemisphere, then full
    re-search + frozen-constant S10 re-run."""
    eng = ctx['eng']
    mask = eng.mask.copy()
    rng = np.random.default_rng(seed_from('pl_v1_sc_null', SCREEN_ID))
    swap_log = []
    for lo, hi in ((0, 180), (180, 360)):
        nodes = np.arange(lo, hi)
        idx = np.triu_indices(len(nodes), 1)
        pairs = [(int(nodes[i]), int(nodes[j])) for i, j in zip(*idx)
                 if mask[nodes[i], nodes[j]] == 1]
        n_swaps = 4 * len(pairs)
        swaps = attempts = 0
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
        swap_log.append({'hemisphere': f'{lo}-{hi}', 'n_edges': len(pairs),
                         'n_swaps': swaps, 'n_attempts': attempts})
    eng2 = P1Engine()
    eng2.load()
    eng2.mask = mask
    m2 = _copy_matrices_with(ctx['matrices'], sc_mask=mask)
    return _screen_rerun(eng2, m2, ctx['uni']), swap_log


# ══════════════════════════════════════════════════════════════════
# feature construction from explicit path lists
# ══════════════════════════════════════════════════════════════════
def base_c4(ctx):
    """L one-hot (20) + 4 endpoint communication metrics — identical for every
    variant; no path information enters C4."""
    uni, eng = ctx['uni'], ctx['eng']
    L_arr = uni['path_length_edges'].values.astype(np.int64)
    onehot = np.zeros((len(uni), 20), dtype=np.float64)
    ok = L_arr <= 20
    onehot[ok] = np.eye(20)[L_arr[ok] - 1]
    si = np.array([eng.r2i[s] for s in uni['seed_roi']])
    ei = np.array([eng.r2i[e] for e in uni['endpoint_roi']])
    ep = np.stack([eng.mats[m][si, ei] for m in METRICS], axis=1)
    return np.hstack([onehot, ep])


def _metric_stack(paths, perm=None):
    """{metric: (K, L)} edge-metric matrices; `perm` (K, L) permutes edge
    POSITIONS within each path, identically across the four metrics."""
    u, v = paths[:, :-1], paths[:, 1:]
    out = {}
    for m in METRICS:
        X = B.ENGINE.mats[m][u, v].astype(np.float64)
        if perm is not None:
            X = np.take_along_axis(X, perm, axis=1)
        out[m] = X
    return out


def blind_from_paths_perm(paths, L, perm):
    """blind_from_paths with permuted edge order (frozen 56-stat math)."""
    K = paths.shape[0]
    if K == 0:
        return np.full(56, np.nan)
    Xs = _metric_stack(paths, perm)
    ppa = np.empty((K, 28), dtype=np.float64)
    if L >= 2:
        z = np.linspace(0.0, 1.0, L)
        zd = z - z.mean()
        zz = float((zd * zd).sum())
    for j, m in enumerate(METRICS):
        X = Xs[m]
        c = 7 * j
        ppa[:, c] = X.mean(axis=1)
        ppa[:, c + 1] = X.std(axis=1, ddof=0)
        ppa[:, c + 2] = X.min(axis=1)
        ppa[:, c + 3] = X.max(axis=1)
        ppa[:, c + 4] = X[:, 0]
        ppa[:, c + 5] = X[:, -1]
        if L >= 2:
            ppa[:, c + 6] = ((X - X.mean(axis=1, keepdims=True))
                             * zd).sum(axis=1) / zz
        else:
            ppa[:, c + 6] = 0.0
    cf = np.zeros(56, dtype=np.float64)
    for j in range(28):
        col = ppa[:, j]
        fin = col[np.isfinite(col)]
        if fin.size > 0:
            cf[2 * j] = np.median(fin)
            cf[2 * j + 1] = (float(np.subtract(*np.percentile(fin, [75, 25])))
                             if fin.size > 1 else 0.0)
        else:
            cf[2 * j] = cf[2 * j + 1] = np.nan
    return cf


def ob_from_paths_perm(paths, L, perm):
    """ob_from_paths with permuted edge order (frozen Legendre math)."""
    K = paths.shape[0]
    if K == 0:
        raise ValueError('empty path set')
    Xs = _metric_stack(paths, perm)
    Bm = B.legendre_basis(L)
    qmax = min(3, L - 1)
    out = {}
    for m in METRICS:
        X = Xs[m]
        beta = np.zeros((K, 4), dtype=np.float64)
        if qmax >= 0:
            Q, R = np.linalg.qr(Bm, mode='reduced')
            beta[:, :qmax + 1] = np.linalg.solve(R, Q.T @ X.T).T
        for q in range(4):
            col = beta[:, q]
            out[f'ob_{m}_p{q}_med'] = float(np.median(col))
            out[f'ob_{m}_p{q}_iqr'] = (
                float(np.subtract(*np.percentile(col, [75, 25])))
                if K > 1 else 0.0)
    return out


def features_from_path_lists(ctx, path_lists, perm_lists=None):
    """(blind 56, ob 32, n_paths) aligned to ctx['uni'] row order.

    perm_lists: optional {sample_id: [edge-position permutation per path]}.
    When absent the frozen builders are called directly, so identity runs are
    bit-identical to the prior pipeline.
    """
    uni = ctx['uni']
    blind = np.empty((len(uni), 56), dtype=np.float64)
    ob = np.empty((len(uni), 32), dtype=np.float64)
    n_paths = np.zeros(len(uni), dtype=np.int64)
    for i, (_, r) in enumerate(uni.iterrows()):
        L = int(r['path_length_edges'])
        sid = r['sample_id']
        pl = path_lists.get(sid, [])
        n_paths[i] = len(pl)
        if not pl:
            blind[i] = np.nan
            ob[i] = np.nan
            continue
        arr = np.asarray(pl, dtype=np.int16)
        if perm_lists is None:
            blind[i] = blind_from_paths(arr, L)
            od = ob_from_paths(arr, L)
        else:
            pm = np.asarray(perm_lists[sid], dtype=np.int64)
            blind[i] = blind_from_paths_perm(arr, L, pm)
            od = ob_from_paths_perm(arr, L, pm)
        ob[i] = [od[c] for c in OB_COLS]
    return blind, ob, n_paths


def build_X(ctx, path_lists, c4=None, perm_lists=None):
    """{'C4','C6','C8R'} design matrices for one path set."""
    c4 = base_c4(ctx) if c4 is None else c4
    blind, ob, n_paths = features_from_path_lists(ctx, path_lists, perm_lists)
    return ({'C4': c4, 'C6': np.hstack([c4, blind]),
             'C8R': np.hstack([c4, ob])}, n_paths)


# ══════════════════════════════════════════════════════════════════
# cross-validation
# ══════════════════════════════════════════════════════════════════
def fold_ids(ctx, rep, fold):
    rd = ctx['fm_sup'][ctx['fm_sup']['repeat'] == rep]
    test_sids = rd.loc[rd['test_fold'] == fold, 'sample_id'].tolist()
    train_sids = rd.loc[rd['test_fold'] != fold, 'sample_id'].tolist()
    tr = np.array([ctx['sid_pos'][s] for s in train_sids], dtype=np.int64)
    te = np.array([ctx['sid_pos'][s] for s in test_sids], dtype=np.int64)
    groups = rd.loc[rd['test_fold'] != fold, 'seed_roi'].values
    return tr, te, test_sids, groups


def run_fold(X, y, tr, te, groups, test_sids, rep, fold, variant):
    res = A4.train_outer_fold(X[tr], y[tr], groups, X[te])
    yt, yp = y[te], res['y_pred']
    mb = A4.metric_block(yt, yp)
    oof = [{'sample_id': sid, 'variant': variant, 'repeat': rep,
            'outer_fold': fold, 'y_true': float(yt[i]), 'y_pred': float(yp[i])}
           for i, sid in enumerate(test_sids)]
    fm = {'variant': variant, 'repeat': rep, 'outer_fold': fold, **mb,
          'best_alpha': res['best_alpha'], 'best_l1_ratio': res['best_l1_ratio'],
          'n_iter_': res['n_iter_'], 'n_warnings': res['n_warnings'],
          'n_train': len(tr), 'n_test': len(te)}
    return oof, fm


def repeat_summary_r2(df, n_rep=5):
    """Mean of the per-repeat pooled R2 (frozen Task C convention)."""
    vals = []
    for rep in range(n_rep):
        sub = df[df['repeat'] == rep]
        if len(sub) == 0:
            continue
        v = A4.metric_block(sub['y_true'].values, sub['y_pred'].values)['r2']
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def seed_map(ctx):
    return ctx['uni'][['sample_id']].merge(
        ctx['fm_sup'][['sample_id', 'seed_roi']].drop_duplicates('sample_id'),
        on='sample_id', how='left')
