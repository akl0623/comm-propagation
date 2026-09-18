#!/usr/bin/env python3
"""
so_lib.py — shared machinery for the S10 optimized validation.

The single methodological change relative to every prior run is the negative
control's PATH SOURCE.  Previously the real arm was C8R built from ALL sampled
paths (~309/cell) while order-shuffle / matched-random controls were built from
the SCREENED registry (~76/cell).  That contrast confounds the manipulation
with the path count and the path population, so it cannot test order- or
path-set-specificity.  Here every control is built from the SAME all-sampled
path source as the real C8R-All arm.

Six strata, four biological hypotheses.  H3/H4 carry a pre-specified primary
(the literal rule) and a labelled sensitivity variant (the relaxed redesign);
the two were frozen in the Round-1/Round-2 screen configs before any modelling.

Everything else — cells, folds, target, preprocessing, 120-candidate
ElasticNet grid — is inherited unchanged from the frozen Task C pipeline.
"""
import os
import sys
import json
import hashlib
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

ROOT = PIPELINE_ROOT
SMCR = os.path.join(ROOT, 'screen_match_confirm_v1')
ICS = os.path.join(ROOT, 'iterative_combined_screen_v1')
B4R = os.path.join(ROOT, 'b4_confirmatory_v1')
PLV = os.path.join(ROOT, 's10_pathlevel_v1')

sys.path.insert(0, os.path.join(SMCR, 'confirmatory', 'code'))
sys.path.insert(0, os.path.join(SMCR, 'candidate_screens', 'code'))
sys.path.insert(0, os.path.join(ICS, 'full_experiment', 'code'))
sys.path.insert(0, os.path.join(PLV, 'code'))

import stage4a_lib as A4                                     # noqa: E402
import bridge_noscreen as B                                  # noqa: E402
from bridge_noscreen import METRICS                          # noqa: E402
import pl_lib as PL                                          # noqa: E402

warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────── output tree
OUT = os.path.join(ROOT, 's10_optimized_v1')
CFG = os.path.join(OUT, 'config')
AUD = os.path.join(OUT, 'audit')
STR = os.path.join(OUT, 'strata')
SYM = os.path.join(OUT, 'symmetric_controls')
SENS = os.path.join(OUT, 'sensitivity')
TBL = os.path.join(OUT, 'tables')
FIG = os.path.join(OUT, 'figures')
REP = os.path.join(OUT, 'reporting')
PSETS = os.path.join(OUT, 'strata', '_path_sets')
FEAT = os.path.join(OUT, 'strata', '_features')
RES = os.path.join(OUT, 'strata', '_results')
for _d in (CFG, AUD, STR, SYM, SENS, TBL, FIG, REP, PSETS, FEAT, RES):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────── frozen inputs
K0_SAMPLE = 512
KSEL = 128
JAC = 0.80
BOOT_SEED = 8769577528861415417        # frozen Task C cluster-bootstrap seed
N_SHUFFLE_REPS = 3
N_BOOT = 10000
N_PERM = 1000
S10_THRESHOLD = 0.41549280879846406
OB_COLS = PL.OB_COLS

# ────────────────────────────────────────────────────────── strata registry
#   screen_id drives the FEATURE seed  SHA256('v1' + screen_id + sample_id)
#   and must match the run that produced the reused results.
STRATA = {
    'H1': {
        'stratum': 'H1',
        'screen_id': 'S10',
        'screen_name': 'B5_plus_highvariance',
        'hypothesis': 'propagation complexity',
        'rule': 'within-path edge-metric variance >= 0.4154928088 (absolute) '
                '+ B5 DDVS',
        'role': 'primary',
        'attrition': os.path.join(ICS, 'smoke_test', 'ROUND2',
                                  'S10_attrition.csv'),
        'registry': os.path.join(ICS, 'smoke_test', 'ROUND2',
                                 'S10_registry.parquet'),
        'reuse_real': os.path.join(ICS, 'full_experiment', 'S10', 'taskC'),
        'reuse_kind': 'ics',
    },
    'H2': {
        'stratum': 'H2',
        'screen_id': 'B4',
        'screen_name': 'B4_baseline',
        'hypothesis': 'structural strength',
        'rule': 'mean path SC >= 0.012645 (frozen 40th pct) + B5 DDVS',
        'role': 'primary',
        'attrition': os.path.join(B4R, 'paths', 'B4_ATTRITION.csv'),
        'registry': os.path.join(B4R, 'paths', 'B4_PATH_REGISTRY.parquet'),
        'reuse_real': os.path.join(B4R, 'taskC'),
        'reuse_kind': 'b4',
    },
    'H3': {
        'stratum': 'H3',
        'screen_id': 'S4',
        'screen_name': 'B5_plus_B2relaxed',
        'hypothesis': 'temporal plausibility',
        'rule': 'every edge velocity in [1,150] m/s + B5 DDVS',
        'role': 'primary',
        'attrition': os.path.join(ICS, 'smoke_test', 'ROUND1',
                                  'S4_attrition.csv'),
        'registry': os.path.join(ICS, 'smoke_test', 'ROUND1',
                                 'S4_registry.parquet'),
        'reuse_real': None,
        'reuse_kind': None,
    },
    'H3s': {
        'stratum': 'H3s',
        'screen_id': 'S14',
        'screen_name': 'B5_plus_medianvelocity',
        'hypothesis': 'temporal plausibility (sensitivity)',
        'rule': 'MEDIAN edge velocity in [1,150] m/s + B5 DDVS',
        'role': 'sensitivity',
        'attrition': os.path.join(ICS, 'smoke_test', 'ROUND2',
                                  'S14_attrition.csv'),
        'registry': os.path.join(ICS, 'smoke_test', 'ROUND2',
                                 'S14_registry.parquet'),
        'reuse_real': None,
        'reuse_kind': None,
    },
    'H4': {
        'stratum': 'H4',
        'screen_id': 'S6',
        'screen_name': 'B5_plus_Yeo7le3',
        'hypothesis': 'network consistency',
        'rule': 'Yeo7 transitions along path <= 3 + B5 DDVS',
        'role': 'primary',
        'attrition': os.path.join(ICS, 'smoke_test', 'ROUND1',
                                  'S6_attrition.csv'),
        'registry': os.path.join(ICS, 'smoke_test', 'ROUND1',
                                 'S6_registry.parquet'),
        'reuse_real': None,
        'reuse_kind': None,
    },
    'H4s': {
        'stratum': 'H4s',
        'screen_id': 'S13',
        'screen_name': 'B5_plus_Yeo7le4',
        'hypothesis': 'network consistency (sensitivity)',
        'rule': 'Yeo7 transitions along path <= 4 + B5 DDVS',
        'role': 'sensitivity',
        'attrition': os.path.join(ICS, 'smoke_test', 'ROUND2',
                                  'S13_attrition.csv'),
        'registry': os.path.join(ICS, 'smoke_test', 'ROUND2',
                                 'S13_registry.parquet'),
        'reuse_real': None,
        'reuse_kind': None,
    },
}
PRIMARY_STRATA = ['H1', 'H2', 'H3', 'H4']
ALL_STRATA = ['H1', 'H2', 'H3', 'H3s', 'H4', 'H4s']
NEW_STRATA = ['H3', 'H3s', 'H4', 'H4s']       # need C4 + C8R-All from scratch

# control arms, all built from the ALL-sampled path source
CONTROL_ARMS = ['C8R-All-OrderShuffle1', 'C8R-All-OrderShuffle2',
                'C8R-All-OrderShuffle3', 'C8R-All-MatchedRandom']
SHUFFLE_ARMS = CONTROL_ARMS[:3]


def log(msg):
    print(msg, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_from(*parts):
    key = '|'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def feature_seed(screen_id, sid):
    """Frozen feature-path seed: SHA256('v1' + screen_id + cell_id)[:8]."""
    return int.from_bytes(
        hashlib.sha256(f'v1{screen_id}{sid}'.encode()).digest()[:8], 'big')


# ══════════════════════════════════════════════════════════════════
# context
# ══════════════════════════════════════════════════════════════════
_ENG = None


def engine():
    global _ENG
    if _ENG is None:
        _ENG = A4.build_engine()
    return _ENG


def load_context(stratum):
    """Engine + this stratum's support universe + folds + target.

    Folds come from the single frozen Task C manifest, restricted to the
    stratum's support — so every stratum uses identical fold assignments.
    """
    spec = STRATA[stratum]
    eng = engine()
    uni_full = pd.read_csv(B.UNI_P)
    fm = pd.read_csv(B.FOLD_P)
    att = pd.read_csv(spec['attrition'])
    support_sids = set(att[att['in_support'].astype(bool)]['sample_id'])
    uni = uni_full[uni_full['sample_id'].isin(support_sids)].reset_index(
        drop=True)
    uni = uni[['sample_id', 'seed_roi', 'endpoint_roi', 'path_length_edges']]
    y = uni['sample_id'].map(
        dict(zip(uni_full['sample_id'],
                 uni_full['y_endpoint'].astype(np.float64)))
    ).values.astype(np.float64)
    return {
        'stratum': stratum,
        'screen_id': spec['screen_id'],
        'eng': eng,
        'uni': uni,
        'uni_full': uni_full,
        'y': y,
        'sid_pos': {sid: i for i, sid in enumerate(uni['sample_id'])},
        'fm_sup': fm[fm['sample_id'].isin(support_sids)].reset_index(drop=True),
        'support_sids': support_sids,
        'attrition': att,
    }


# ══════════════════════════════════════════════════════════════════
# path sets
# ══════════════════════════════════════════════════════════════════
def pset_path(stratum, name):
    return os.path.join(PSETS, f'{stratum}__{name}.parquet')


def save_path_set(stratum, name, path_lists):
    rows = []
    for sid, pl in path_lists.items():
        for k, nodes in enumerate(pl):
            rows.append({'sample_id': sid, 'path_index': k,
                         'L': len(nodes) - 1,
                         'roi_id_sequence': '|'.join(str(int(x))
                                                     for x in nodes)})
    p = pset_path(stratum, name)
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def load_path_set(stratum, name):
    df = pd.read_parquet(pset_path(stratum, name))
    out = {}
    for sid, g in df.groupby('sample_id', sort=False):
        out[sid] = [[int(x) for x in s.split('|')]
                    for s in g.sort_values('path_index')['roi_id_sequence']]
    return out


def build_all_sampled(ctx):
    """K0=512 ranks over the cell's FULL simple-path space with the stratum's
    frozen feature seed — reproduces that stratum's C8R-All bit-for-bit."""
    eng, screen_id = ctx['eng'], ctx['screen_id']
    out = {}
    for _, r in ctx['uni'].iterrows():
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        ranks = eng.sample_ranks(si, ei, L, K0_SAMPLE,
                                 feature_seed(screen_id, r['sample_id']))
        arr = eng.unrank_cell(si, ei, L, ranks)
        out[r['sample_id']] = [[int(x) for x in row] for row in arr]
    return out


def _valid(eng, cand):
    """SC edge present + all four metrics finite on every consecutive pair.

    Strict temporal ordering is deliberately NOT required: DAG nodes are
    latency-sorted, so requiring it would make the identity permutation the
    only legal ordering and collapse the control.
    """
    for i in range(1, len(cand)):
        u, v = cand[i - 1], cand[i]
        if eng.mask[u, v] != 1:
            return False
        for m in METRICS:
            if not np.isfinite(eng.mats[m][u, v]):
                return False
    return True


def build_order_shuffle_all(ctx, all_paths, rep):
    """SYMMETRIC order control: interior ROIs of each ALL-SAMPLED path randomly
    permuted, seed and endpoint fixed, edge multiset otherwise untouched.
    Same path source, same count as the real C8R-All arm."""
    eng, screen_id = ctx['eng'], ctx['screen_id']
    out = {}
    n_total = n_fallback = n_identity = 0
    for sid, pl in all_paths.items():
        rows = []
        for k, nodes in enumerate(pl):
            n_total += 1
            if len(nodes) <= 3:                # L<=2: no reorderable interior
                rows.append(list(nodes))
                n_identity += 1
                continue
            rng = np.random.default_rng(
                seed_from('so_v1_order_shuffle_all', screen_id, rep, sid, k))
            placed = None
            for _ in range(20):
                mids = list(nodes[1:-1])
                rng.shuffle(mids)
                cand = [nodes[0]] + mids + [nodes[-1]]
                if cand == list(nodes):
                    continue
                if _valid(eng, cand):
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


def path_mean_sc(nodes, sc):
    n = np.asarray(nodes, dtype=np.int64)
    return float(np.mean(sc[n[:-1], n[1:]]))


def build_matched_random_all(ctx, all_paths, sc):
    """SYMMETRIC path-set control: an INDEPENDENT draw from the same full
    simple-path space, at the same per-cell count and L, greedily matched to
    the real all-sampled set on per-path mean SC.

    DEGENERACY WARNING, recorded per cell in the diagnostics: when the cell's
    total path count T <= K0 the real arm is already the COMPLETE enumeration
    of the path space, so no alternative path set exists and the control is
    identical to the real arm by construction.  Only cells with T > K0 carry
    information for this contrast.
    """
    eng, screen_id = ctx['eng'], ctx['screen_id']
    out = {}
    diag = []
    for _, r in ctx['uni'].iterrows():
        sid = r['sample_id']
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        target = all_paths[sid]
        n_t = len(target)
        T = eng.counts(si, ei, L)
        degenerate = bool(T <= K0_SAMPLE)
        pool_size = min(T, max(2 * K0_SAMPLE, n_t))
        ranks = eng.sample_ranks(si, ei, L, pool_size,
                                 seed_from('so_v1_matched_random_all',
                                           screen_id, sid))
        pool_nodes = [[int(x) for x in row]
                      for row in eng.unrank_cell(si, ei, L, ranks)]
        tgt_sc = np.array([path_mean_sc(p, sc) for p in target])
        pool_sc = np.array([path_mean_sc(p, sc) for p in pool_nodes])
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
        diag.append({'sample_id': sid, 'L': L, 'T_total_paths': int(T),
                     'n_target': n_t, 'n_pool': len(pool_nodes),
                     'n_matched': len(sel),
                     'degenerate_full_enumeration': degenerate,
                     'mean_sc_target': (float(np.mean(tgt_sc))
                                        if n_t else np.nan),
                     'mean_sc_matched': (
                         float(np.mean([path_mean_sc(p, sc) for p in sel]))
                         if sel else np.nan),
                     'overlap_with_target': len(
                         {tuple(p) for p in sel} & {tuple(p) for p in target})})
    return out, pd.DataFrame(diag)


# ══════════════════════════════════════════════════════════════════
# features
# ══════════════════════════════════════════════════════════════════
def base_c4(ctx):
    """L one-hot (20) + 4 endpoint communication metrics.  Carries no path
    information, so it is identical across every arm within a stratum."""
    uni, eng = ctx['uni'], ctx['eng']
    L_arr = uni['path_length_edges'].values.astype(np.int64)
    onehot = np.zeros((len(uni), 20), dtype=np.float64)
    ok = L_arr <= 20
    onehot[ok] = np.eye(20)[L_arr[ok] - 1]
    si = np.array([eng.r2i[s] for s in uni['seed_roi']])
    ei = np.array([eng.r2i[e] for e in uni['endpoint_roi']])
    ep = np.stack([eng.mats[m][si, ei] for m in METRICS], axis=1)
    return np.hstack([onehot, ep])


def build_c8r(ctx, path_lists, c4):
    """C4 + 32-dim Legendre ordered basis built from the given path lists."""
    _, ob, n_paths = PL.features_from_path_lists(ctx, path_lists)
    return np.hstack([c4, ob]), n_paths


def feat_path(stratum, arm):
    return os.path.join(FEAT, f'X_{stratum}__{arm}.npy')


def build_arm_X(ctx, arm, c4):
    """Cached design matrix for one arm of one stratum."""
    stratum = ctx['stratum']
    cache = feat_path(stratum, arm)
    npc = os.path.join(FEAT, f'npaths_{stratum}__{arm}.npy')
    if os.path.exists(cache):
        return np.load(cache), np.load(npc)
    if arm == 'C4':
        X, n_paths = c4, np.zeros(len(ctx['uni']), dtype=np.int64)
    else:
        pset = {'C8R-All': 'all_sampled',
                'C8R-All-OrderShuffle1': 'order_shuffle_all_rep1',
                'C8R-All-OrderShuffle2': 'order_shuffle_all_rep2',
                'C8R-All-OrderShuffle3': 'order_shuffle_all_rep3',
                'C8R-All-MatchedRandom': 'matched_random_all'}[arm]
        X, n_paths = build_c8r(ctx, load_path_set(stratum, pset), c4)
    np.save(cache, X)
    np.save(npc, n_paths)
    return X, n_paths


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


def run_fold(X, y, tr, te, groups, test_sids, rep, fold, arm):
    res = A4.train_outer_fold(X[tr], y[tr], groups, X[te])
    yt, yp = y[te], res['y_pred']
    mb = A4.metric_block(yt, yp)
    oof = [{'sample_id': sid, 'arm': arm, 'repeat': rep, 'outer_fold': fold,
            'y_true': float(yt[i]), 'y_pred': float(yp[i])}
           for i, sid in enumerate(test_sids)]
    fm = {'arm': arm, 'repeat': rep, 'outer_fold': fold, **mb,
          'best_alpha': res['best_alpha'], 'best_l1_ratio': res['best_l1_ratio'],
          'n_iter_': res['n_iter_'], 'n_warnings': res['n_warnings'],
          'n_train': len(tr), 'n_test': len(te)}
    return oof, fm


def repeat_r2(df):
    """Mean of the per-repeat pooled R2 (frozen Task C convention)."""
    vals = [A4.metric_block(g['y_true'].values, g['y_pred'].values)['r2']
            for _, g in df.groupby('repeat')]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def seed_map(ctx):
    return ctx['uni'][['sample_id']].merge(
        ctx['fm_sup'][['sample_id', 'seed_roi']].drop_duplicates('sample_id'),
        on='sample_id', how='left')


def oof_path(stratum, arm):
    return os.path.join(RES, f'OOF_{stratum}__{arm}.parquet')


def fm_path(stratum, arm):
    return os.path.join(RES, f'FOLD_METRICS_{stratum}__{arm}.csv')


# ══════════════════════════════════════════════════════════════════
# inference — paired seed-cluster bootstrap + paired sign-flip permutation
# ══════════════════════════════════════════════════════════════════
def paired_bootstrap(a_df, b_df, smap, b=N_BOOT):
    """Frozen cluster-bootstrap math; the SAME cluster resample is applied to
    both arms, so the difference is paired."""
    a = a_df.copy()
    a['model'] = 'a'
    bb = b_df.copy()
    bb['model'] = 'b'
    df = pd.concat([a, bb], ignore_index=True).merge(
        smap, on='sample_id', how='inner')
    stats = {}
    for m in ('a', 'b'):
        sub = df[df['model'] == m]
        gg = sub.groupby(['seed_roi', 'repeat'])
        y = gg['y_true'].apply(np.asarray)
        p = gg['y_pred'].apply(np.asarray)
        stats[m] = pd.DataFrame({
            'n': gg.size(), 'Sy': y.apply(np.nansum),
            'Syy': y.apply(lambda v: np.nansum(v * v)),
            'SSE': y.combine(p, lambda x_, z: np.nansum((x_ - z) ** 2)),
        }).reset_index()
    clusters = sorted(set().union(*[set(s['seed_roi']) for s in stats.values()]))
    n_clu = len(clusters)
    clu_idx = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, n_clu, size=(b, n_clu))
    mults = np.zeros((b, n_clu), dtype=np.int64)
    for j in range(b):
        np.add.at(mults[j], draws[j], 1)
    boot = {}
    for m in ('a', 'b'):
        s = stats[m].copy()
        s['ci'] = s['seed_roi'].map(clu_idx).values
        acc = np.zeros(b)
        n_finite = np.zeros(b)
        for rep in sorted(s['repeat'].unique()):
            sub = s[s['repeat'] == rep]
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
    delta = boot['a'] - boot['b']
    d = delta[np.isfinite(delta)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean(d > 0)), int(d.size))


def _perm_frame(a_df, b_df, smap):
    a = a_df.merge(smap, on='sample_id', how='left')
    cols = ['sample_id', 'repeat', 'outer_fold', 'y_true', 'seed_roi']
    a = a[cols + ['y_pred']].rename(columns={'y_pred': 'pa'})
    bb = b_df[['sample_id', 'repeat', 'outer_fold', 'y_pred']].rename(
        columns={'y_pred': 'pb'})
    m = a.merge(bb, on=['sample_id', 'repeat', 'outer_fold'], how='inner')
    return m.sort_values(['repeat', 'outer_fold', 'sample_id']).reset_index(
        drop=True)


def paired_permutation(a_df, b_df, smap, tag, n_perm=N_PERM):
    """Exchange the two arms' predictions at random within (seed_roi, repeat)
    strata, holding (sample, fold) identity; two-sided p.

    Vectorised form of the frozen sign-flip test.  A flip is constant within a
    (seed_roi, repeat) group, and repeat-level R2 is 1 - SSE/SST with SST fixed
    by y_true, so a permutation only needs the GROUP sums of the two arms'
    squared errors.  Algebraically identical to flipping row by row and
    recomputing R2, but O(n_groups) instead of O(n) per permutation.
    """
    m = _perm_frame(a_df, b_df, smap)
    y = m['y_true'].values
    ea = (y - m['pa'].values) ** 2
    eb = (y - m['pb'].values) ** 2
    reps = np.sort(m['repeat'].unique())
    # per-repeat SST (constant across permutations)
    sst = {}
    for r in reps:
        yr = y[m['repeat'].values == r]
        sst[r] = float(((yr - yr.mean()) ** 2).sum()) if len(yr) >= 3 else np.nan
    # group-level error sums, in a deterministic (seed_roi, repeat) order
    gkeys = sorted(m.groupby(['seed_roi', 'repeat']).indices.items(),
                   key=lambda kv: (str(kv[0][0]), int(kv[0][1])))
    n_g = len(gkeys)
    g_rep = np.array([int(k[1]) for k, _ in gkeys])
    Sa = np.array([ea[idx].sum() for _, idx in gkeys])
    Sb = np.array([eb[idx].sum() for _, idx in gkeys])
    rep_of = {r: (g_rep == r) for r in reps}

    def _r2_mean(sse_by_group_a, sse_by_group_b):
        """(mean repeat R2 for arm a, same for arm b) given per-group SSEs."""
        va, vb = [], []
        for r in reps:
            s = rep_of[r]
            if not np.isfinite(sst[r]) or sst[r] <= 0:
                continue
            va.append(1 - sse_by_group_a[s].sum() / sst[r])
            vb.append(1 - sse_by_group_b[s].sum() / sst[r])
        return (float(np.mean(va)) if va else np.nan,
                float(np.mean(vb)) if vb else np.nan)

    r2a0, r2b0 = _r2_mean(Sa, Sb)
    obs = r2a0 - r2b0
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        g = np.random.default_rng(seed_from('so_v1_signflip', tag, k))
        flip = g.integers(0, 2, size=n_g).astype(bool)
        ka = np.where(flip, Sb, Sa)
        kb = np.where(flip, Sa, Sb)
        ra, rb = _r2_mean(ka, kb)
        null[k] = ra - rb
    p_two = 2 * min(float((null <= obs).mean()), float((null >= obs).mean()))
    return obs, null, min(1.0, p_two)


def paired_permutation_reference(a_df, b_df, smap, tag, n_perm=50):
    """Row-by-row reference implementation, used only to validate the fast
    path.  Recomputes R2 through the frozen metric_block on every draw."""
    m = _perm_frame(a_df, b_df, smap)
    obs = (repeat_r2(m.rename(columns={'pa': 'y_pred'}))
           - repeat_r2(m.rename(columns={'pb': 'y_pred'})))
    gkeys = sorted(m.groupby(['seed_roi', 'repeat']).indices.items(),
                   key=lambda kv: (str(kv[0][0]), int(kv[0][1])))
    pa, pb = m['pa'].values, m['pb'].values
    mk = m[['repeat', 'y_true']].copy()
    null = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        g = np.random.default_rng(seed_from('so_v1_signflip', tag, k))
        picks = g.integers(0, 2, size=len(gkeys)).astype(bool)
        flip = np.zeros(len(m), dtype=bool)
        for (_, idx), pick in zip(gkeys, picks):
            if pick:
                flip[idx] = True
        mk['y_pred'] = np.where(flip, pb, pa)
        r2a = repeat_r2(mk)
        mk['y_pred'] = np.where(flip, pa, pb)
        null[k] = r2a - repeat_r2(mk)
    return obs, null


def repeats_positive(a_df, b_df):
    out = []
    for rep in sorted(set(a_df['repeat'])):
        ra = repeat_r2(a_df[a_df['repeat'] == rep])
        rb = repeat_r2(b_df[b_df['repeat'] == rep])
        out.append((ra - rb) if (np.isfinite(ra) and np.isfinite(rb))
                   else np.nan)
    return out


def pooled_shuffle_arm(stratum, arms):
    """A shuffled arm is a SINGLE prediction vector (the mean over the three
    independent realisations), never an ensemble of three models."""
    parts = [pd.read_parquet(oof_path(stratum, v)) for v in arms]
    return (pd.concat(parts)
            .groupby(['sample_id', 'repeat', 'outer_fold'])
            .agg(y_true=('y_true', 'first'), y_pred=('y_pred', 'mean'))
            .reset_index())


def contrast(a_df, b_df, smap, name, do_perm=True, tag=None):
    """`name` labels the contrast; `tag` seeds the permutation draws and must
    be unique across strata (defaults to `name`)."""
    tag = tag or name
    lo, hi, frac, nb = paired_bootstrap(a_df, b_df, smap)
    ra, rb = repeat_r2(a_df), repeat_r2(b_df)
    reps = repeats_positive(a_df, b_df)
    row = {'contrast': name, 'r2_a': ra, 'r2_b': rb,
           'point_estimate': ra - rb,
           'bootstrap_ci_lo': lo, 'bootstrap_ci_hi': hi,
           'bootstrap_frac_gt_0': frac, 'n_boot_finite': nb,
           'n_repeats_positive': int(np.sum([v > 0 for v in reps
                                             if np.isfinite(v)])),
           'n_repeats': int(np.sum(np.isfinite(reps))),
           'repeats': json.dumps([None if not np.isfinite(v) else float(v)
                                  for v in reps])}
    if do_perm:
        obs, null, p = paired_permutation(a_df, b_df, smap, tag)
        row['permutation_p_two_sided'] = p
        row['perm_obs'] = obs
    else:
        row['permutation_p_two_sided'] = None
        row['perm_obs'] = None
    return row


# ══════════════════════════════════════════════════════════════════
# S10 screen application (Phase 02): vectorised B5 validity + variance stat
# ══════════════════════════════════════════════════════════════════
from run_screens import gen_candidates, make_matrices                # noqa: E402


def build_matrices(eng):
    return make_matrices(eng, B.RAW, B.BI, os.path.join(SMCR, 'config'))


def cell_path_stats(eng, matrices, si, ei, L, screen_id, sid, K0=K0_SAMPLE):
    """(nodes, b5_valid, variance_stat) for one cell's candidate paths.

    Vectorised equivalent of ps_screens.b5_hard_validity + s10_component.
    L<2 paths are variance-exempt and returned with stat = NaN.
    """
    paths = gen_candidates(eng, si, ei, L, screen_id, sid, K0)
    if not paths:
        return (np.empty((0, L + 1), dtype=np.int64), np.zeros(0, bool),
                np.zeros(0))
    nodes = np.asarray([p.nodes for p in paths], dtype=np.int64)
    K = nodes.shape[0]
    u, v = nodes[:, :-1], nodes[:, 1:]
    srt = np.sort(nodes, axis=1)
    ok = ~(np.diff(srt, axis=1) == 0).any(axis=1)          # no repeated ROI
    ok &= (matrices.sc_mask[u, v] == 1).all(axis=1)        # SC edge present
    vals = matrices.onset[si][nodes].astype(np.float64).copy()
    vals[:, 0] = 0.0
    ok &= (np.diff(vals, axis=1) > 0).all(axis=1)          # strictly temporal
    for m in METRICS:
        ok &= np.isfinite(getattr(matrices, m)[u, v]).all(axis=1)
    if L < 2:
        return nodes, ok, np.full(K, np.nan)
    stat = np.zeros(K, dtype=np.float64)
    for m in METRICS:
        stat += np.var(getattr(matrices, m)[u, v], axis=1, ddof=0) / 4.0
    return nodes, ok, stat


def s10_support(eng, matrices, uni, threshold, screen_id='S10',
                K0=K0_SAMPLE, collect_stats=False):
    """Apply S10 at `threshold` and return the attrition frame.

    Support counts paths passing every reject rule PRE-DDVS (frozen
    convention): L=1 needs >=1, L>=2 needs >=5.  DDVS affects only the
    screened-path registry, which this analysis never uses — C8R-All is built
    from ALL sampled paths — so it is deliberately not applied here.
    """
    rows, pool = [], []
    for _, r in uni.iterrows():
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        nodes, ok, stat = cell_path_stats(eng, matrices, si, ei, L,
                                          screen_id, r['sample_id'], K0)
        if L >= 2:
            acc = ok & (stat >= threshold)
            if collect_stats:
                pool.append(stat[ok])
        else:
            acc = ok
        n_acc = int(acc.sum())
        rows.append({'sample_id': r['sample_id'], 'L': L,
                     'seed_roi': r['seed_roi'],
                     'endpoint_roi': r['endpoint_roi'],
                     'n_candidate': int(nodes.shape[0]),
                     'n_b5_valid': int(ok.sum()), 'n_accepted': n_acc,
                     'in_support': bool(n_acc >= (1 if L == 1 else 5))})
    att = pd.DataFrame(rows)
    return (att, np.concatenate(pool) if pool else np.zeros(0)) \
        if collect_stats else att


def context_from_support(uni_full, att, screen_id, y_col, fold_manifest,
                         label):
    """A minimal ctx for arbitrary universes (Task A as well as Task C)."""
    sup = set(att[att['in_support'].astype(bool)]['sample_id'])
    uni = uni_full[uni_full['sample_id'].isin(sup)].reset_index(drop=True)
    uni = uni[['sample_id', 'seed_roi', 'endpoint_roi', 'path_length_edges']]
    y = uni['sample_id'].map(dict(zip(uni_full['sample_id'],
                                      uni_full[y_col].astype(np.float64))))
    return {
        'stratum': label, 'screen_id': screen_id, 'eng': engine(),
        'uni': uni, 'uni_full': uni_full,
        'y': y.values.astype(np.float64),
        'sid_pos': {s: i for i, s in enumerate(uni['sample_id'])},
        'fm_sup': fold_manifest[
            fold_manifest['sample_id'].isin(sup)].reset_index(drop=True),
        'support_sids': sup, 'attrition': att,
    }


def grouped_fold_manifest(uni, repeat_seeds=(20260727, 20260728, 20260729,
                                            20260730, 20260731), n_folds=5):
    """5 repeats x seed-grouped folds, mirroring the frozen Task C manifest.

    Within a repeat the seed list is shuffled with that repeat's frozen seed,
    then whole seeds are assigned largest-first to the currently lightest fold
    (the GroupKFold rule), so no seed ever spans two folds.
    """
    sizes = uni.groupby('seed_roi').size()
    rows = []
    for rep, rs in enumerate(repeat_seeds):
        seeds = np.array(sorted(sizes.index))
        np.random.default_rng(rs).shuffle(seeds)
        order = sorted(seeds, key=lambda s: (-int(sizes[s]),
                                             list(seeds).index(s)))
        load = np.zeros(n_folds, dtype=np.int64)
        assign = {}
        for s in order:
            f = int(np.argmin(load))
            assign[s] = f
            load[f] += int(sizes[s])
        for _, r in uni.iterrows():
            rows.append({'repeat': rep, 'repeat_seed': rs,
                         'sample_id': r['sample_id'],
                         'seed_roi': r['seed_roi'],
                         'test_fold': assign[r['seed_roi']]})
    return pd.DataFrame(rows)


def _write_pset(path, path_lists):
    """Path-set I/O by absolute path (Task A lives outside strata/)."""
    rows = [{'sample_id': sid, 'path_index': k, 'L': len(n) - 1,
             'roi_id_sequence': '|'.join(str(int(x)) for x in n)}
            for sid, pl in path_lists.items() for k, n in enumerate(pl)]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _read_pset(path):
    df = pd.read_parquet(path)
    return {sid: [[int(x) for x in s.split('|')]
                  for s in g.sort_values('path_index')['roi_id_sequence']]
            for sid, g in df.groupby('sample_id', sort=False)}
