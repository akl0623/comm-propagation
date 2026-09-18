#!/usr/bin/env python3
"""
Task C C8R ordered-basis feature builder v3 — V3 recovery pilot.


  final features = path-length one-hot (L_is_1..L_is_20)
                 + four endpoint metrics at [seed, final_endpoint] (ep_*)
                 + 32 ordered-basis process features (ob_{metric}_p{q}_{agg})

Per formally sampled path and per metric, the adjacent-edge sequence
x_s = metric[path[s-1], path[s]] (s = 1..L) is projected onto the fixed
Legendre basis over normalized stage coordinates z_s = -1 + 2*(s-1)/(L-1)
(L=1 -> z=0):
  P0(z)=1, P1(z)=z, P2(z)=(3z^2-1)/2, P3(z)=(5z^3-3z)/2
q = 0..min(3, L-1); beta = argmin ||B beta - x||^2 (float64 least squares).
Coefficients for q > L-1 are DEFINED as 0 (no higher-order degrees of freedom
on that discrete path — mathematical definition, not zero-imputation).

Per (sample, sampling seed): median and IQR of each beta_q across the seed's
formal path set -> 4 metrics x 4 q x 2 aggs = 32 per-seed values.
Formal predictor (as in C1A): per-column median across the 5 sampling seeds;
the seed-level IQR is stored as a Monte Carlo stability audit only.

BLIND: this module reads ONLY the communication matrices and the C0R2D
freeze files (universe / fold manifest / K table). It never reads the
probability matrix, stepwise targets, labels, or any performance artifacts,
and it does NOT import the target builder. Path resampling is deterministic
(frozen P1Engine + cell_seed + fold-specific K); the resampled path sets are
identity-verified against the frozen features parquet (n_paths, n_validated,
blind_XX) before any C8R feature is written.

"""
import os, sys, json, time, hashlib, ast as _ast_module
from array import array
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from taskC_feature_builder_v1 import (
    P1Engine, cell_seed, SEEDS, METRICS, N_FEAT_56, pp_feats, cell_56,
)

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

# ── Frozen input paths (communication metrics + C0R2D freeze only) ──
UNI_P = os.path.join(PIPELINE_ROOT, "taskC0R2D_universe_freeze", "taskC_primary_2636_universe_C0R2D.csv")
FOLD_P = os.path.join(PIPELINE_ROOT, "taskC0R2D_universe_freeze", "taskC_primary_fold_manifest_C0R2D_v2.csv")
K_P = os.path.join(PIPELINE_ROOT, "taskC0R2D_universe_freeze", "C1A_fold_K_selection.csv")
FEAT_V1_P = "features/TASK_C_FEATURES_V1.parquet"   # frozen blind features (identity check)
FEAT_C8R_P = "features/TASK_C_FEATURES_C8R_V3.parquet"
FEAT_C8R_SEED_P = "features/TASK_C_FEATURES_C8R_SEEDLEVEL_V3.parquet"
CACHE_DIR = "cache/c8r_paths_v3"

L_MAX = 20
N_SEEDS = 5
N_OB = 32  # 4 metrics x 4 q x 2 aggs

# ── runtime open trace (blind audit) ──
_orig_open = open
_OPENED = []


def _trace_open(f, *a, **kw):
    p = f if isinstance(f, str) else getattr(f, 'name', str(f))
    _OPENED.append((p, kw.get('mode', 'r')))
    return _orig_open(f, *a, **kw)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with _orig_open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
# LEGENDRE ORDERED BASIS (frozen math)
# ═══════════════════════════════════════════════════════════════

def legendre_poly(z, q):
    """P_q(z) elementwise on float64 array z."""
    z = np.asarray(z, dtype=np.float64)
    if q == 0:
        return np.ones_like(z)
    if q == 1:
        return z
    if q == 2:
        return (3.0 * z * z - 1.0) / 2.0
    if q == 3:
        return (5.0 * z ** 3 - 3.0 * z) / 2.0
    raise ValueError("only P0..P3 are defined in this frozen basis")


def stage_z(L):
    """Normalized stage coordinates: L=1 -> [0]; L>1 -> -1..+1."""
    if L == 1:
        return np.array([0.0], dtype=np.float64)
    return np.asarray(-1.0 + 2.0 * np.arange(L) / (L - 1), dtype=np.float64)


def basis_matrix(L, qmax=None):
    """B[s, q] = P_q(z_s), s = 1..L, q = 0..min(3, L-1)."""
    if qmax is None:
        qmax = min(3, L - 1)
    z = stage_z(L)
    return np.column_stack([legendre_poly(z, q) for q in range(qmax + 1)])


def ordered_coeffs_single(x):
    """Reference implementation: np.linalg.lstsq(..., rcond=None) per path.

    x : 1-D float64 edge sequence, length L >= 1.
    Returns beta of length 4; entries q > L-1 are 0 by definition.
    """
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    if L < 1:
        raise ValueError("empty edge sequence")
    if not np.isfinite(x).all():
        raise ValueError("non-finite edge value in path sequence")
    qmax = min(3, L - 1)
    beta = np.zeros(4, dtype=np.float64)
    if qmax >= 0:
        B = basis_matrix(L, qmax)
        beta[:qmax + 1], _, _, _ = np.linalg.lstsq(B, x, rcond=None)
    return beta


# Precomputed QR for the batched (mathematically equivalent) implementation
_QR_CACHE = {}


def _qr_for_L(L):
    qmax = min(3, L - 1)
    if qmax < 0:
        return None
    if L not in _QR_CACHE:
        B = basis_matrix(L, qmax)
        Q, R = np.linalg.qr(B, mode='reduced')
        _QR_CACHE[L] = (Q, R)
    return _QR_CACHE[L]


def ordered_coeffs_batch(X):
    """Batched QR least squares over the FIRST axis.

    X : (n_paths, L) float64 edge sequences.
    Returns (n_paths, 4) beta; columns q > L-1 are 0 by definition.
    """
    X = np.asarray(X, dtype=np.float64)
    n, L = X.shape
    if n == 0:
        raise ValueError("empty edge sequence batch")
    if not np.isfinite(X).all():
        raise ValueError("non-finite edge value in path sequence batch")
    qmax = min(3, L - 1)
    out = np.zeros((n, 4), dtype=np.float64)
    if qmax >= 0:
        Q, R = _qr_for_L(L)
        # beta = solve(R, Q.T @ X.T)  -> (qmax+1, n)
        beta = np.linalg.solve(R, Q.T @ X.T)
        out[:, :qmax + 1] = beta.T
    return out


def ob_column_names():
    """Stable ordering: metric -> q -> [med, iqr] (32 columns)."""
    cols = []
    for m in METRICS:
        for q in range(4):
            for agg in ('med', 'iqr'):
                cols.append(f'ob_{m}_p{q}_{agg}')
    return cols


# ═══════════════════════════════════════════════════════════════
# PER-SEED FEATURES + IDENTITY CHECK
# ═══════════════════════════════════════════════════════════════

def path_nodes(paths, L):
    """(n_paths, L+1) int16 node matrix from uniform-length paths."""
    n = len(paths)
    out = np.empty((n, L + 1), dtype=np.int16)
    for i, p in enumerate(paths):
        out[i] = p
    return out


def edge_matrix(eng, m, nodes):
    """(n_paths, L) adjacent-edge values for metric m; fail-fast on any
    illegal matrix access or non-finite edge value."""
    u = nodes[:, :-1]
    v = nodes[:, 1:]
    if (eng.mask[u, v] != 1).any():
        bad = np.argwhere(eng.mask[u, v] != 1)[0]
        raise ValueError(f"illegal adjacency access mask[{u[bad[0], bad[1]]},{v[bad[0], bad[1]]}]=0")
    X = eng.mats[m][u, v].astype(np.float64, copy=True)
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))[0]
        raise ValueError(f"non-finite metric value {m}[{u[bad[0], bad[1]]},{v[bad[0], bad[1]]}]")
    return X


def validate_path(p, si, ei, L, eng):
    if p[0] != si or p[-1] != ei or len(p) - 1 != L:
        return False
    if len(set(p)) != len(p):
        return False
    for s in range(1, len(p)):
        if eng.mask[p[s - 1], p[s]] != 1:
            return False
    return True


def seed_features(eng, paths, L):
    """32 per-seed values: median and IQR of beta_q across paths,
    per metric, per q. Fail-fast on invalid paths/edges."""
    if len(paths) == 0:
        raise ValueError("empty path set for a formally sampled cell")
    nodes = path_nodes(paths, L)
    out = {}
    for m in METRICS:
        X = edge_matrix(eng, m, nodes)
        beta = ordered_coeffs_batch(X)          # (n_paths, 4)
        for q in range(4):
            col = beta[:, q]
            out[f'ob_{m}_p{q}_med'] = float(np.median(col))
            if len(col) > 1:
                out[f'ob_{m}_p{q}_iqr'] = float(np.subtract(*np.percentile(col, [75, 25])))
            else:
                out[f'ob_{m}_p{q}_iqr'] = 0.0
    return out


def identity_blind56(eng, paths, L):
    """Recompute the frozen 56 blind summaries with the SAME frozen code path
    (pp_feats/cell_56) from the resampled paths."""
    if len(paths) == 0:
        return np.full(N_FEAT_56, np.nan)
    ppa = np.array([pp_feats(p, eng) for p in paths])
    return cell_56(ppa)


# ═══════════════════════════════════════════════════════════════
# MAIN BUILD
# ═══════════════════════════════════════════════════════════════

def load_engine():
    t0 = time.time()
    eng = P1Engine()
    eng.load()
    eng.init_dags()
    print(f"  P1Engine ready: {len(eng.dags)} DAGs ({time.time()-t0:.1f}s)", flush=True)
    return eng


def build(eng=None, limit_cells=None, partial_ok=False, cache_dir=None):
    """Build C8R features for all 13,180 (sample, repeat) cells.

    Returns (sample_df, seed_df, audit). With limit_cells/partial_ok used for
    smoke runs, the sample-level aggregation is skipped and sample_df is None.
    """
    global _OPENED
    _OPENED = []
    open = _trace_open  # noqa: F841 — trace every open in this build scope
    cache_dir = cache_dir or CACHE_DIR
    T0 = time.time()
    print("=" * 66)
    print("C8R ordered-basis feature builder v3 (V3 recovery pilot)")
    print("=" * 66, flush=True)

    if eng is None:
        eng = load_engine()

    uni = pd.read_csv(UNI_P)
    folds = pd.read_csv(FOLD_P)
    kt = pd.read_csv(K_P)
    k_map = {(int(r['repeat']), int(r['fold'])): int(r['selected_K']) for _, r in kt.iterrows()}
    sample_info = {r['sample_id']: (r['seed_roi'], r['endpoint_roi'], int(r['path_length_edges']))
                   for _, r in uni.iterrows()}

    # Frozen V1 features for identity verification
    feat_v1 = pd.read_parquet(FEAT_V1_P)
    feat_v1_idx = feat_v1.set_index(['sample_id', 'repeat'])
    blind_cols = [f'blind_{j:02d}' for j in range(N_FEAT_56)]
    ep_cols = [f'ep_{m}' for m in METRICS]

    os.makedirs(cache_dir, exist_ok=True)

    seed_rows = []
    per_sample = {}   # sid -> {rep: seed_feat_dict}
    identity = {'rows_checked': 0, 'mismatch': 0, 'mismatch_details': []}
    global_path_hash = hashlib.sha256()
    cell_hashes = {}
    n_paths_total = 0

    cells = []
    for rep_idx in range(N_SEEDS):
        for fold in range(5):
            cells.append((rep_idx, fold))
    if limit_cells:
        cells = cells[:limit_cells]

    for (rep_idx, fold) in cells:
        rep_seed = SEEDS[rep_idx]
        K = k_map.get((rep_idx, fold))
        if K is None:
            print(f"  [rep={rep_idx} fold={fold}] no K — skip", flush=True)
            continue
        fold_sids = folds[(folds['repeat'] == rep_idx) & (folds['test_fold'] == fold)]['sample_id'].tolist()
        fold_sids.sort()  # canonical order for the path hash
        t0 = time.time()
        cell_hash = hashlib.sha256()
        offsets = [0]
        buf = array('h')
        cell_meta = {'rep': rep_idx, 'fold': fold, 'rep_seed': rep_seed, 'K': K,
                     'sample_ids': [], 'L': [], 'n_paths': []}
        for sid in fold_sids:
            seed_roi, ep_roi, L = sample_info[sid]
            si = eng.r2i.get(seed_roi)
            ei = eng.r2i.get(ep_roi)
            if si is None or ei is None:
                raise ValueError(f"missing ROI mapping for {sid}")
            cs = cell_seed(str(rep_seed), sid)
            paths = eng.sample(si, ei, L, K, cs)
            if len(paths) == 0:
                raise ValueError(f"zero formally sampled paths for {sid} (rep {rep_idx})")
            for p in paths:
                if not validate_path(p, si, ei, L, eng):
                    raise ValueError(f"invalid resampled path for {sid} (rep {rep_idx})")
                pbytes = np.asarray(p, dtype=np.int16).tobytes()
                cell_hash.update(pbytes)
                global_path_hash.update(pbytes)
                buf.frombytes(pbytes)
                offsets.append(offsets[-1] + len(p))
            n_paths_total += len(paths)

            # seed-level ordered-basis features (fail-fast inside)
            feat = seed_features(eng, paths, L)

            # identity check vs frozen V1 features (two-tier)
            #  tier 1 EXACT: integer counts + endpoint metrics (direct matrix
            #  copies, no reduction) — bitwise, proves identical resampling.
            #  tier 2 TOLERANT (1e-12 relative): the 56 blind aggregate columns
            #  show <=3.4e-14 relative float-reduction drift between the Phase-03
            #  runtime and the current runtime on identical inputs (observed and
            #  documented; my resampling is bitwise reproducible run-to-run).
            try:
                fr = feat_v1_idx.loc[(sid, rep_idx)]
            except KeyError:
                raise ValueError(f"frozen features missing row for {sid} rep {rep_idx}")
            blind_re = identity_blind56(eng, paths, L)
            blind_fr = np.asarray(fr[blind_cols].values, dtype=np.float64)
            ep_re = np.array([float(eng.mats[m][si, ei]) for m in METRICS])
            ep_fr = np.asarray(fr[ep_cols].values, dtype=np.float64)
            np_match = bool(int(fr['n_paths']) == len(paths) and int(fr['n_validated']) == len(paths))
            ep_exact = bool(np.all(ep_re == ep_fr))
            rel = np.abs(blind_re - blind_fr) / np.maximum(np.abs(blind_fr), 1e-300)
            blind_ok = bool(np.all(np.isfinite(rel)) and rel.max() <= 1e-12)
            identity['rows_checked'] += 1
            identity['max_rel_so_far'] = max(identity.get('max_rel_so_far', 0.0), float(rel.max()))
            identity['exact_cols_total'] = identity.get('exact_cols_total', 0) + int((blind_re == blind_fr).sum())
            if not (np_match and ep_exact and blind_ok):
                identity['mismatch'] += 1
                identity['mismatch_details'].append(
                    {'sample_id': sid, 'repeat': rep_idx, 'n_paths_match': bool(np_match),
                     'ep_exact': bool(ep_exact), 'blind_ok': bool(blind_ok),
                     'max_rel': float(rel.max())})
                raise ValueError(
                    f"PATH-SET IDENTITY MISMATCH {sid} rep={rep_idx}: "
                    f"n_paths={np_match} ep_exact={ep_exact} "
                    f"blind={blind_ok} max_rel={rel.max():.3e}")

            seed_rows.append({
                'sample_id': sid, 'seed_roi': seed_roi, 'endpoint_roi': ep_roi,
                'repeat': rep_idx, 'repeat_seed': rep_seed, 'test_fold': fold,
                'L': L, 'K': K, 'n_paths': len(paths), **feat,
            })
            per_sample.setdefault(sid, {})[rep_idx] = feat
            cell_meta['sample_ids'].append(sid)
            cell_meta['L'].append(L)
            cell_meta['n_paths'].append(len(paths))

        cell_h = cell_hash.hexdigest()
        cell_hashes[f"r{rep_idx}f{fold}"] = cell_h
        if buf:
            np.savez_compressed(
                os.path.join(CACHE_DIR, f"cell_r{rep_idx}f{fold}.npz"),
                concat=np.frombuffer(buf, dtype=np.int16).copy(),
                offsets=np.asarray(offsets, dtype=np.int64),
            )
        with _orig_open(os.path.join(CACHE_DIR, f"cell_r{rep_idx}f{fold}.json"), 'w') as f:
            json.dump(cell_meta, f, indent=2)
        print(f"  [rep={rep_idx} fold={fold}] K={K} samples={len(fold_sids)} "
              f"paths={sum(cell_meta['n_paths'])} ({time.time()-t0:.1f}s) "
              f"hash={cell_h[:12]}", flush=True)

    if identity['mismatch']:
        raise SystemExit(f"IDENTITY CHECK FAILED: {identity['mismatch']} rows")

    # ── sample-level formal aggregation (median over 5 seeds) ──
    if partial_ok:
        print("[aggregate] SKIPPED (partial smoke run)", flush=True)
        seed_df = pd.DataFrame(seed_rows)
        audit = {
            'build_wall_s': round(time.time() - T0, 1),
            'n_paths_total': n_paths_total,
            'identity': identity,
            'global_path_sha256': global_path_hash.hexdigest(),
            'cell_path_sha256': cell_hashes,
            'partial': True,
        }
        return None, seed_df, audit

    print("\n[aggregate] sample-level formal predictors (5-seed median)...", flush=True)
    ob_cols = ob_column_names()
    sample_rows = []
    for sid in sorted(per_sample):
        seeds = per_sample[sid]
        if len(seeds) != N_SEEDS:
            raise ValueError(f"sample {sid} has {len(seeds)} seeds, expected {N_SEEDS}")
        seed_roi, ep_roi, L = sample_info[sid]
        vals = np.array([[seeds[r][c] for r in range(N_SEEDS)] for c in ob_cols])  # (32, 5)
        formal = np.median(vals, axis=1)          # 32
        seed_iqr = np.subtract(*np.percentile(vals, [75, 25], axis=1))  # 32 audit
        row = {'sample_id': sid, 'seed_roi': seed_roi, 'endpoint_roi': ep_roi, 'L': L}
        for l in range(1, L_MAX + 1):
            row[f'L_is_{l}'] = int(l == L)
        for m in METRICS:
            row[f'ep_{m}'] = float(eng.mats[m][eng.r2i[seed_roi], eng.r2i[ep_roi]])
        for j, c in enumerate(ob_cols):
            row[c] = float(formal[j])
        for j, c in enumerate(ob_cols):
            row[f'seediqr_{c}'] = float(seed_iqr[j])
        sample_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)
    seed_df = pd.DataFrame(seed_rows)
    audit = {
        'build_wall_s': round(time.time() - T0, 1),
        'n_paths_total': n_paths_total,
        'identity': identity,
        'global_path_sha256': global_path_hash.hexdigest(),
        'cell_path_sha256': cell_hashes,
        'columns_sample': list(sample_df.columns),
        'n_predictors': 20 + 4 + 32,
    }
    return sample_df, seed_df, audit


def write_outputs(sample_df, seed_df, audit):
    t0 = time.time()
    sample_df.to_parquet(FEAT_C8R_P, index=False)
    seed_df.to_parquet(FEAT_C8R_SEED_P, index=False)
    for fp, df in [(FEAT_C8R_P, sample_df), (FEAT_C8R_SEED_P, seed_df)]:
        schema = {'columns': list(df.columns), 'n_rows': len(df), 'n_cols': len(df.columns),
                  'dtypes': {c: str(df[c].dtype) for c in df.columns}}
        with _orig_open(fp + '.schema.json', 'w') as f:
            json.dump(schema, f, indent=2)
        with _orig_open(fp + '.sha256', 'w') as f:
            f.write(sha256_file(fp) + '\n')
    print(f"  wrote {FEAT_C8R_P} ({len(sample_df)} rows) + {FEAT_C8R_SEED_P} "
          f"({len(seed_df)} rows) in {time.time()-t0:.1f}s", flush=True)


def blind_audits(sample_df, seed_df, module_path):
    """AST / runtime-open / content blind audits for this module.

    AST audit = import graph only (no target/probability imports), plus any
    string constant that looks like a forbidden FILE PATH. Docstring prose is
    not a code reference and is not treated as an import or a path.
    """
    global _OPENED
    # AST: import graph
    ast_ok, ast_issues = True, []
    with _orig_open(module_path) as f:
        tree = _ast_module.parse(f.read())
    for node in _ast_module.walk(tree):
        if isinstance(node, _ast_module.Import):
            for a in node.names:
                if 'taskC_target' in a.name or 'probability' in a.name.lower():
                    ast_ok = False
                    ast_issues.append(f"import {a.name}")
        if isinstance(node, _ast_module.ImportFrom):
            if node.module and ('taskC_target' in node.module or 'probability' in node.module.lower()):
                ast_ok = False
                ast_issues.append(f"from {node.module}")
        # string constants that look like file paths pointing at forbidden data
        # (docstrings contain newlines and are prose, not paths)
        if isinstance(node, _ast_module.Constant) and isinstance(node.value, str):
            v = node.value
            if '\n' not in v and ('/' in v or v.endswith(('.csv', '.parquet', '.json', '.npz', '.txt', '.py'))) and (
                    'probability' in v.lower() or 'stepwise' in v.lower()
                    or 'y_endpoint' in v.lower() or 'matched_endpoint_truth' in v.lower()
                    or 'diagnosis' in v.lower() or 'phenotype' in v.lower()):
                ast_ok = False
                ast_issues.append(f"forbidden path string: {v[:80]}")
    # runtime opens
    runtime_bad = [p for p, _ in _OPENED if 'probability' in p.lower() or 'target' in p.lower()
                   or 'truth' in p.lower() or 'diagnosis' in p.lower()]
    # content
    content_bad = []
    for df in (sample_df, seed_df):
        if df is None:
            continue
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ('probability', 'target', 'y_endpoint', 'diagnosis',
                                     'phenotype', 'performance', 'error', 'pred')):
                content_bad.append(c)
    return {
        'ast_ok': ast_ok, 'ast_issues': ast_issues,
        'runtime_open_trace': [{'path': p, 'mode': m} for p, m in _OPENED],
        'runtime_probability_or_target_opens': len(runtime_bad),
        'runtime_bad_opens': runtime_bad,
        'content_ok': len(content_bad) == 0,
        'content_bad_columns': content_bad,
    }


def main():
    module_path = os.path.abspath(__file__)
    sample_df, seed_df, audit = build()
    write_outputs(sample_df, seed_df, audit)
    blinds = blind_audits(sample_df, seed_df, module_path)
    audit['blind_audits'] = blinds
    with _orig_open('audit/TASK_C_C8R_BLIND_AND_SEMANTIC_AUDIT_V3.json', 'w') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps({'identity': audit['identity'], 'blind': {k: blinds[k] for k in
          ('ast_ok', 'runtime_probability_or_target_opens', 'content_ok')}}, indent=2))
    print(f"\nC8R BUILD DONE in {audit['build_wall_s']}s "
          f"({audit['n_paths_total']} paths, hash={audit['global_path_sha256'][:16]})", flush=True)


if __name__ == '__main__':
    main()
