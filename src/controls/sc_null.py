#!/usr/bin/env python3
"""
Task C Phase 07 SC-null v1 — degree-preserving within-hemisphere connected
double-edge-swap null graphs, the four communication matrices, and the
matched (summary/ordered) features. Frozen by

Vendored authoritative formulas (verbatim, formula-identity verified against
the frozen matrices to 5.0e-7 = the %.6f save rounding):
  - rout_efficiency / distance_wei_floyd / retrieve_shortest_path /
    search_information : Brain Connectivity Toolbox (bct) algorithms,
    copied verbatim with attribution (the only edit: retrieved paths are
    raveled for scalar indexing, semantics unchanged).
  - navigation_efficiency   : project script
    'count communication measure.py' (Seguin 2018 PNAS greedy
    navigation on tract LENGTH guided by nodal distance D), copied verbatim.
  - calculate_cmy           : degree-normalized communicability
    CMY = expm(D S D), D = diag(1/sqrt(strength)), copied verbatim.
    Allowed optimization: symmetric eigendecomposition Q exp(Lambda) Q^T with
    frozen acceptance (max |diff| vs expm <= 1e-8 on toy + real + nulls 1-3),
    automatic fallback to scipy.linalg.expm for ALL nulls on any failure.

Null graphs: per-hemisphere connected double-edge swaps on the corrected
symmetric binary SC; (tract_strength, tract_length) as inseparable attribute
pairs randomly reassigned within hemisphere; cross-hemisphere edges frozen.
Seed rule: SHA-256(f'{master_seed}|{hemisphere}|{null_id}')[:8] big-endian.
"""
import os, sys, json, hashlib, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from scipy.linalg import expm

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

# ── frozen inputs ──
SC_P = os.path.join(PIPELINE_ROOT, "taskC0R2C_symmetric_SC_and_P1_freeze", "averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv")
LEN_P = os.path.join(PIPELINE_ROOT, "taskC0R2C_symmetric_SC_and_P1_freeze", "averageConnectivity_tractLength_0.25density_symmetric_C0R2C.csv")
DIST_P = os.path.join(PIPELINE_ROOT, "raw_inputs", "fsaverage_parcel_distance_matrix.csv")
MASTER_SEED = 20260727
N = 360
HEMI = [0] * 180 + [1] * 180  # 0=left, 1=right (C0R2C ROI order)

GRAPH_DIR = 'cache/phase07_v5/sc_null_graphs'
MATRIX_DIR = 'cache/phase07_v5/communication_matrices'


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════
# VENDORED BCT ALGORITHMS (verbatim; attribution: Avena-Koenigsberger &
# Goni, Brain Connectivity Toolbox, python port by Roan LaPlante)
# ═══════════════════════════════════════════════════════════════

def distance_wei_floyd(adjacency, transform=None):
    # it is important not to do these transformations safely, to allow infinity
    if transform is not None:
        with np.errstate(divide='ignore'):
            if transform == 'log':
                SPL = -np.log(adjacency)
            elif transform == 'inv':
                SPL = 1 / adjacency
            else:
                raise ValueError("Unexpected transform type. Only 'log' and "
                                 "'inv' are accepted")
    else:
        SPL = adjacency.copy().astype('float')
        SPL[SPL == 0] = np.inf

    n = adjacency.shape[1]
    hops = np.array(adjacency != 0).astype('float')
    Pmat = np.repeat(np.atleast_2d(np.arange(0, n)), n, 0)

    for k in range(n):
        i2k_k2j = np.repeat(SPL[:, [k]], n, 1) + np.repeat(SPL[[k], :], n, 0)
        path = SPL > i2k_k2j
        i, j = np.where(path)
        hops[path] = hops[i, k] + hops[k, j]
        Pmat[path] = Pmat[i, k]
        SPL = np.min(np.stack([SPL, i2k_k2j], 2), 2)

    I = np.eye(n) > 0
    SPL[I] = 0
    hops[I], Pmat[I] = 0, 0
    return SPL, hops, Pmat


def retrieve_shortest_path(s, t, hops, Pmat):
    path_length = hops[s, t]
    if path_length != 0:
        path = np.zeros((int(path_length + 1), 1), dtype='int')
        path[0] = s
        for ind in range(1, len(path)):
            s = Pmat[s, t]
            path[ind] = s
    else:
        path = []
    return path


def rout_efficiency(D, transform=None):
    n = len(D)
    Erout, _, _ = distance_wei_floyd(D, transform=transform)
    with np.errstate(divide='ignore'):
        Erout = 1 / Erout
    np.fill_diagonal(Erout, 0)
    GErout = (np.sum(Erout[np.where(np.logical_not(np.isnan(Erout)))]) /
              (n ** 2 - n))
    return GErout, Erout


def search_information(adjacency, transform=None, has_memory=False):
    N = len(adjacency)
    flag_triu = bool(np.allclose(adjacency, adjacency.T))
    T = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)
    _, hops, Pmat = distance_wei_floyd(adjacency, transform)
    SI = np.zeros((N, N))
    SI[np.eye(N) > 0] = np.nan
    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = np.asarray(retrieve_shortest_path(i, j, hops, Pmat)).ravel()
                lp = len(path) - 1
                if flag_triu:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        pr_step_bk = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[int(path[0]), int(path[1])]
                            pr_step_bk[lp - 1] = T[int(path[lp]), int(path[lp - 1])]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[int(path[z]), int(path[z + 1])] / \
                                    (1 - T[int(path[z - 1]), int(path[z])])
                                pr_step_bk[lp - z - 1] = T[int(path[lp - z]), int(path[lp - z - 1])] / \
                                    (1 - T[int(path[lp - z + 1]), int(path[lp - z])])
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[int(path[z]), int(path[z + 1])]
                                pr_step_bk[z] = T[int(path[z + 1]), int(path[z])]
                        prob_sp_ff = np.prod(pr_step_ff)
                        prob_sp_bk = np.prod(pr_step_bk)
                        SI[i, j] = -np.log2(prob_sp_ff)
                        SI[j, i] = -np.log2(prob_sp_bk)
                else:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        for z in range(lp):
                            pr_step_ff[z] = T[int(path[z]), int(path[z + 1])]
                        SI[i, j] = -np.log2(np.prod(pr_step_ff))
                    else:
                        SI[i, j] = np.inf
    return SI


# ═══════════════════════════════════════════════════════════════
# VENDORED navigation_efficiency (verbatim, Seguin 2018 PNAS)
# ═══════════════════════════════════════════════════════════════

def navigation_efficiency(L, D, max_hops=None):
    N = L.shape[0]
    if max_hops is None:
        max_hops = N
    PL_bin = np.zeros((N, N))
    PL_wei = np.zeros((N, N))
    PL_dis = np.zeros((N, N))
    paths = [[[] for _ in range(N)] for _ in range(N)]
    E = np.zeros((N, N))
    successful_pairs = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                curr_node = i
                last_node = curr_node
                target = j
                path = [curr_node]
                pl_bin = 0
                pl_wei = 0
                pl_dis = 0
                success = True
                while curr_node != target:
                    neighbors = np.where(L[curr_node, :] != 0)[0]
                    if len(neighbors) == 0:
                        success = False
                        break
                    neighbor_dists = D[j, neighbors]
                    min_index = np.argmin(neighbor_dists)
                    next_node = neighbors[min_index]
                    if next_node == last_node or pl_bin >= max_hops:
                        success = False
                        break
                    path.append(next_node)
                    pl_bin += 1
                    pl_wei += L[curr_node, next_node]
                    pl_dis += D[curr_node, next_node]
                    last_node = curr_node
                    curr_node = next_node
                if success and curr_node == target:
                    PL_bin[i, j] = pl_bin
                    PL_wei[i, j] = pl_wei
                    PL_dis[i, j] = pl_dis
                    paths[i][j] = path.copy()
                    E[i, j] = 1 / pl_wei if pl_wei > 0 else 0
                    successful_pairs += 1
                else:
                    PL_bin[i, j] = np.inf
                    PL_wei[i, j] = np.inf
                    PL_dis[i, j] = np.inf
                    paths[i][j] = []
                    E[i, j] = 0
            else:
                paths[i][j] = []
                PL_bin[i, j] = np.inf
                PL_wei[i, j] = np.inf
                PL_dis[i, j] = np.inf
                E[i, j] = 0
    valid_pairs = N * (N - 1)
    failed = np.sum(PL_bin == np.inf) - N
    sr = 1 - failed / valid_pairs
    non_diag_mask = ~np.eye(N, dtype=bool)
    nav_eff_global = np.mean(E[non_diag_mask])
    return sr, PL_bin, PL_wei, PL_dis, paths, E, nav_eff_global


def calculate_cmy(W):
    N = W.shape[0]
    s = np.sum(W, axis=1)
    s_safe = np.where(s > 0, s, 1)
    D = np.diag(1 / np.sqrt(s_safe))
    W_prime = D @ W @ D
    CMY = expm(W_prime)
    return CMY


def calculate_cmy_eig(W):
    """Numerically equivalent symmetric-eigendecomposition implementation
    (frozen acceptance vs expm; automatic fallback at the caller)."""
    N = W.shape[0]
    s = np.sum(W, axis=1)
    s_safe = np.where(s > 0, s, 1)
    D = np.diag(1 / np.sqrt(s_safe))
    W_prime = D @ W @ D
    w, Q = np.linalg.eigh((W_prime + W_prime.T) / 2.0)
    return (Q * np.exp(w)[None, :]) @ Q.T


# ═══════════════════════════════════════════════════════════════
# NULL GRAPH GENERATION (frozen seed rule)
# ═══════════════════════════════════════════════════════════════

def null_seed(null_id, hemisphere):
    msg = f"{MASTER_SEED}|{hemisphere}|{null_id}".encode('utf-8')
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], 'big')


def load_frozen_inputs():
    S = pd.read_csv(SC_P, header=None).values.astype(np.float64)
    L = pd.read_csv(LEN_P, header=None).values.astype(np.float64)
    D = pd.read_csv(DIST_P, index_col=0).values.astype(np.float64)
    assert S.shape == (360, 360) and L.shape == (360, 360) and D.shape == (360, 360)
    assert not np.isnan(S).any() and not np.isnan(L).any()
    assert np.allclose(S, S.T) and np.allclose(L, L.T)
    return S, L, D


def hemi_nodes(h):
    return np.arange(h * 180, h * 180 + 180)


def hemi_subgraph(adj, h):
    nodes = hemi_nodes(h)
    return adj[np.ix_(nodes, nodes)]


def hemi_edges(adj, h):
    """Undirected edge list (i<j) of hemisphere h in GLOBAL indices."""
    nodes = hemi_nodes(h)
    sub = hemi_subgraph(adj, h)
    tri = np.triu(np.ones_like(sub, dtype=bool), 1)
    ii, jj = np.where(tri & (sub > 0))
    return np.stack([nodes[ii], nodes[jj]], axis=1)  # (E, 2)


def is_connected(adj, h):
    sub = hemi_subgraph(adj, h)
    if sub.shape[0] == 0:
        return False
    seen = np.zeros(sub.shape[0], dtype=bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        nb = np.where(sub[u] > 0)[0]
        for v in nb:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    return bool(seen.all())


def _connected(sub):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n_comp, _ = connected_components(csr_matrix(sub), directed=False)
    return n_comp == 1


def _apply_swap(sub, edges, i1, i2, a, b, c, d, x1, y1, x2, y2):
    sub[a, b] = sub[b, a] = False
    sub[c, d] = sub[d, c] = False
    sub[x1, y1] = sub[y1, x1] = True
    sub[x2, y2] = sub[y2, x2] = True
    edges[i1] = (x1, y1)
    edges[i2] = (x2, y2)


def _revert_swap(sub, edges, i1, i2, a, b, c, d, x1, y1, x2, y2):
    sub[a, b] = sub[b, a] = True
    sub[c, d] = sub[d, c] = True
    sub[x1, y1] = sub[y1, x1] = False
    sub[x2, y2] = sub[y2, x2] = False
    edges[i1] = (a, b)
    edges[i2] = (c, d)


def _swap_loop(sub, rng, E, target, max_attempts):
    """Deterministic BATCHED double-edge-swap loop on a 180x180 boolean
    adjacency (frozen V5 optimization: non-conflicting valid swaps of one
    batch are applied together; if a batch ever disconnects the hemisphere,
    the batch is replayed sequentially with exact per-swap connectivity
    checks — the connectedness guarantee is exact).
    Returns (sub, accepted, attempts, rej_selfloop, rej_multi, rej_disconnect).
    """
    edges = [(int(u), int(v)) for u in range(180) for v in range(u + 1, 180)
             if sub[u, v]]
    assert len(edges) == E
    accepted = attempts = 0
    rej_selfloop = rej_multi = rej_disconnect = rej_batch_conflict = 0
    BATCH = 2000
    while accepted < target and attempts < max_attempts:
        b = min(BATCH, max_attempts - attempts)
        cands = []
        for _ in range(b):
            attempts += 1
            i1 = rng.randrange(E)
            i2 = rng.randrange(E)
            if i1 == i2:
                continue
            a, b_ = edges[i1]
            c, d = edges[i2]
            if len({a, b_, c, d}) < 4:
                rej_selfloop += 1
                continue
            if rng.random() < 0.5:
                x1, y1, x2, y2 = a, c, b_, d
            else:
                x1, y1, x2, y2 = a, d, b_, c
            if sub[x1, y1] or sub[x2, y2]:
                rej_multi += 1
                continue
            cands.append((i1, i2, a, b_, c, d, x1, y1, x2, y2))
        # apply-time exact re-checks (candidates are validated against the
        # CURRENT graph state; no node-set filter needed): stale edge-list
        # positions, multi-edges and self-loops are skipped
        applied = []
        for cand in cands:
            i1, i2, a, b_, c, d, x1, y1, x2, y2 = cand
            if edges[i1] != (a, b_) or edges[i2] != (c, d):
                rej_batch_conflict += 1
                continue
            if sub[x1, y1] or sub[x2, y2]:
                rej_multi += 1
                continue
            if x1 == y1 or x2 == y2:
                rej_selfloop += 1
                continue
            _apply_swap(sub, edges, i1, i2, a, b_, c, d, x1, y1, x2, y2)
            applied.append(cand)
        if applied and not _connected(sub):
            # rare: replay sequentially with exact per-swap checks
            for cand in reversed(applied):
                _revert_swap(sub, edges, *cand[1:])
            for cand in applied:
                i1, i2, a, b_, c, d, x1, y1, x2, y2 = cand
                _apply_swap(sub, edges, i1, i2, a, b_, c, d, x1, y1, x2, y2)
                if _connected(sub):
                    accepted += 1
                else:
                    _revert_swap(sub, edges, i1, i2, a, b_, c, d, x1, y1, x2, y2)
                    rej_disconnect += 1
        else:
            accepted += len(applied)
    return sub, accepted, attempts, rej_selfloop, rej_multi, rej_disconnect


def build_null_graph_full(null_id, S0, L0, target_mult=4, max_attempt_mult=20):
    import random
    S = S0.copy()
    L = L0.copy()
    mask = (S0 > 0).astype(np.int8)
    cross = int((mask[:180, 180:] > 0).sum() + (mask[180:, :180] > 0).sum())
    audit = {'null_id': null_id, 'hemispheres': {}, 'cross_edges_frozen': cross}
    for h in (0, 1):
        nodes = hemi_nodes(h)
        sub = hemi_subgraph(mask, h).astype(bool).copy()
        E = int(sub[np.triu_indices_from(sub, 1)].sum())
        rng = random.Random(null_seed(null_id, h))
        pairs = []
        for u in range(180):
            for v in range(u + 1, 180):
                if sub[u, v]:
                    pairs.append((float(S0[nodes[u], nodes[v]]),
                                  float(L0[nodes[u], nodes[v]])))
        target = target_mult * E
        max_attempts = max_attempt_mult * target
        sub, accepted, attempts, rs, rm, rd = _swap_loop(
            sub, rng, E, target, max_attempts)
        rng.shuffle(pairs)
        # reassign the (strength, length) multiset to the rewired edges
        k = 0
        S[np.ix_(nodes, nodes)] = 0.0
        L[np.ix_(nodes, nodes)] = 0.0
        for u in range(180):
            for v in range(u + 1, 180):
                if sub[u, v]:
                    sval, lval = pairs[k]
                    k += 1
                    S[nodes[u], nodes[v]] = S[nodes[v], nodes[u]] = sval
                    L[nodes[u], nodes[v]] = L[nodes[v], nodes[u]] = lval
        assert k == len(pairs) == E
        audit['hemispheres'][str(h)] = {
            'E': E, 'target_swaps': target, 'attempts': attempts,
            'accepted': accepted, 'acceptance_rate': accepted / max(attempts, 1),
            'rej_selfloop': rs, 'rej_multi': rm, 'rej_disconnect': rd,
            'reached_target': accepted >= target}
    return S, L, audit


def validate_null_graph(S_null, L_null, S0, L0):
    """All frozen per-null checks; raises on any violation."""
    checks = {}
    assert S_null.shape == (360, 360)
    assert np.allclose(S_null, S_null.T), "symmetry"
    mask0 = (S0 > 0)
    maskn = (S_null > 0)
    # degree vectors exact (per hemisphere)
    deg0 = {h: hemi_subgraph(mask0, h).sum(axis=1) for h in (0, 1)}
    degn = {h: hemi_subgraph(maskn, h).sum(axis=1) for h in (0, 1)}
    for h in (0, 1):
        assert np.array_equal(deg0[h], degn[h]), f"degree vector hemisphere {h}"
        assert int(maskn[np.ix_(hemi_nodes(h), hemi_nodes(h))].sum()) == \
            int(mask0[np.ix_(hemi_nodes(h), hemi_nodes(h))].sum()), "edge count"
    # no self loops
    assert not np.any(np.diag(maskn)), "self loop"
    # no multi-edge by construction (binary)
    # connectivity per hemisphere
    for h in (0, 1):
        assert is_connected(maskn, h), f"disconnected hemisphere {h}"
    # strength-length pair multiset exact per hemisphere
    for h in (0, 1):
        nodes = hemi_nodes(h)
        p0 = sorted((S0[nodes[u], nodes[v]], L0[nodes[u], nodes[v]])
                    for u in range(180) for v in range(u + 1, 180) if mask0[nodes[u], nodes[v]])
        pn = sorted((S_null[nodes[u], nodes[v]], L_null[nodes[u], nodes[v]])
                    for u in range(180) for v in range(u + 1, 180) if maskn[nodes[u], nodes[v]])
        assert p0 == pn, f"pair multiset hemisphere {h}"
    # cross-hemisphere edges frozen
    assert np.array_equal(S_null[:180, 180:], S0[:180, 180:]), "cross edges changed"
    checks['degree_exact'] = True
    checks['edge_count_exact'] = True
    checks['symmetric'] = True
    checks['connected_both_hemispheres'] = True
    checks['pair_multiset_exact'] = True
    checks['no_selfloop_multi'] = True
    checks['no_new_cross_hemisphere_edges'] = True
    return checks


# ═══════════════════════════════════════════════════════════════
# FOUR MATRICES PER NULL (computed exactly once)
# ═══════════════════════════════════════════════════════════════

def compute_four_matrices(S, L, D, use_eig=True):
    """routing (L), navigation (L,D), search information (S, log),
    normalized communicability (S). Returns dict of float64 matrices."""
    t0 = time.time()
    _, Erout = rout_efficiency(L, transform=None)
    sr, PL_bin, PL_wei, PL_dis, paths, E, nav_eff = navigation_efficiency(L, D, max_hops=360)
    SI = search_information(S, transform='log')
    if use_eig:
        CMY = calculate_cmy_eig(S)
    else:
        CMY = calculate_cmy(S)
    return {'rout': Erout, 'nav': E, 'search': SI, 'comm': CMY,
            'wall_s': time.time() - t0, 'nav_success_ratio': sr,
            'nav_eff_global': nav_eff}


def round6(M):
    return np.round(np.asarray(M, dtype=np.float64), 6)


def matrix_finite_summary(M):
    off = ~np.eye(360, dtype=bool)
    return {'shape': list(M.shape), 'finite_offdiag': int(np.isfinite(M[off]).sum()),
            'nan_total': int(np.isnan(M).sum()),
            'min': float(np.nanmin(M)), 'max': float(np.nanmax(M))}


# ═══════════════════════════════════════════════════════════════
# AUDIT / CACHE IO
# ═══════════════════════════════════════════════════════════════

def save_graph(S_null, L_null, audit, null_id):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    ghash = sha256_bytes(np.ascontiguousarray(
        (S_null > 0).astype(np.int8)).tobytes())
    audit['graph_hash'] = ghash
    np.savez_compressed(os.path.join(GRAPH_DIR, f'null_{null_id:03d}.npz'),
                        S=S_null, L=L_null)
    json.dump(audit, open(os.path.join(GRAPH_DIR, f'null_{null_id:03d}.json'), 'w'),
              indent=2)
    return ghash


def save_matrices(mats, null_id, graph_hash, formula_hash):
    os.makedirs(MATRIX_DIR, exist_ok=True)
    rounded = {k: round6(v) for k, v in mats.items()
               if k not in ('wall_s', 'nav_success_ratio', 'nav_eff_global')}
    np.savez_compressed(os.path.join(MATRIX_DIR, f'null_{null_id:03d}.npz'), **rounded)
    meta = {
        'null_id': null_id, 'graph_hash': graph_hash, 'formula_hash': formula_hash,
        'finite': {k: matrix_finite_summary(v) for k, v in rounded.items()},
        'wall_s': mats['wall_s'], 'nav_success_ratio': mats['nav_success_ratio'],
        'nav_eff_global': mats['nav_eff_global'],
        'communicability_impl': 'eig_QexpQt_or_expm_per_v5',
        'content_hash': sha256_file(os.path.join(MATRIX_DIR, f'null_{null_id:03d}.npz')),
    }
    json.dump(meta, open(os.path.join(MATRIX_DIR, f'null_{null_id:03d}.json'), 'w'),
              indent=2)
    return meta


if __name__ == '__main__':
    print("taskC_phase07_sc_null_v1 — library module (Phase 07 V5)")
