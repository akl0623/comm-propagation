"""P1 DAG engine — strict full-precision latency, corrected symmetric SC."""
import numpy as np, hashlib, random
from collections import defaultdict
from functools import lru_cache

def p1_count_for_seed(stim_idx, sc_mask, lat_matrix, hemisphere_indices=None):
    if hemisphere_indices is None:
        hemisphere_indices = list(range(0,180)) if stim_idx<180 else list(range(180,360))
    nodes = [(stim_idx, 0.0)]
    for ri in hemisphere_indices:
        if ri == stim_idx: continue
        tv = lat_matrix[stim_idx, ri]
        if np.isnan(tv) or tv <= 0: continue
        nodes.append((ri, float(tv)))
    if len(nodes) <= 1: return {}
    nodes.sort(key=lambda x: x[1])
    node_order = [n[0] for n in nodes]
    node_lat = {n[0]: n[1] for n in nodes}
    out_edges = defaultdict(list)
    for ui in node_order:
        for vi in node_order:
            if node_lat[vi] <= node_lat[ui]: continue
            if sc_mask[ui, vi]: out_edges[ui].append(vi)
        out_edges[ui] = sorted(set(out_edges[ui]))
    if not out_edges: return {}
    result = {}
    for ep in node_order:
        if ep == stim_idx: continue
        node_to_len = {n: {} for n in node_order}
        node_to_len[ep][0] = 1
        ep_idx = node_order.index(ep)
        for pos in range(ep_idx, -1, -1):
            u = node_order[pos]
            for v in out_edges.get(u, []):
                for Lm, cnt in node_to_len[v].items():
                    L = Lm + 1
                    node_to_len[u][L] = node_to_len[u].get(L, 0) + cnt
        for L, cnt in node_to_len[stim_idx].items():
            if L >= 1 and cnt > 0: result[(stim_idx, ep, L)] = cnt
    return result

def unrank_path(rank, stim_idx, endpoint, length, sc_mask, lat_matrix, hemisphere_indices=None):
    if hemisphere_indices is None:
        hemisphere_indices = list(range(0,180)) if stim_idx<180 else list(range(180,360))
    nodes = [(stim_idx, 0.0)]
    for ri in hemisphere_indices:
        if ri == stim_idx: continue
        tv = lat_matrix[stim_idx, ri]
        if np.isnan(tv) or tv <= 0: continue
        nodes.append((ri, float(tv)))
    nodes.sort(key=lambda x: x[1])
    node_order = [n[0] for n in nodes]
    node_lat = {n[0]: n[1] for n in nodes}
    out_edges = defaultdict(list)
    for ui in node_order:
        for vi in node_order:
            if node_lat[vi] <= node_lat[ui]: continue
            if sc_mask[ui, vi]: out_edges[ui].append(vi)
        out_edges[ui] = sorted(set(out_edges[ui]))
    
    # Boundary check
    counts = p1_count_for_seed(stim_idx, sc_mask, lat_matrix, hemisphere_indices)
    total = int(counts.get((stim_idx, endpoint, length), 0))
    if rank < 0 or rank >= total:
        raise ValueError(f"Rank {rank} out of bounds [0, {total})")
    
    @lru_cache(maxsize=None)
    def count_to(u, remaining):
        if remaining == 0: return 1 if u == endpoint else 0
        if u not in out_edges: return 0
        total_c = 0
        for v in out_edges[u]: total_c += count_to(v, remaining - 1)
        return total_c
    
    path = [stim_idx]; current = stim_idx; remaining = length; rank_rem = rank
    while remaining > 0:
        candidates = out_edges.get(current, [])
        counts_v = [count_to(v, remaining - 1) for v in candidates]
        total_v = sum(counts_v)
        if total_v == 0: raise ValueError(f"No completions")
        if rank_rem >= total_v: raise ValueError(f"Rank {rank_rem} >= {total_v}")
        cum = 0; chosen = None
        for v, c in zip(candidates, counts_v):
            if rank_rem < cum + c: chosen = v; break
            cum += c
        path.append(chosen); current = chosen; remaining -= 1; rank_rem -= cum
    return path

def sample_paths(stim_idx, endpoint, length, sc_mask, lat_matrix, n_samples, cell_seed, hemisphere_indices=None):
    counts = p1_count_for_seed(stim_idx, sc_mask, lat_matrix, hemisphere_indices)
    total = int(counts.get((stim_idx, endpoint, length), 0))
    if total == 0: return []
    n = min(n_samples, total)
    rng = random.Random(cell_seed)
    if total <= 2**62:
        ranks = sorted(rng.sample(range(total), n))
    else:
        ranks_set = set()
        for i in range(total - n, total):
            t = rng.randrange(i + 1)
            if t not in ranks_set: ranks_set.add(t)
            else: ranks_set.add(i)
        ranks = sorted(ranks_set)
    return [unrank_path(int(r), stim_idx, endpoint, length, sc_mask, lat_matrix, hemisphere_indices) for r in ranks]

def derive_cell_seed(global_seed, sample_id):
    key = f"{global_seed}|{sample_id}".encode()
    h = hashlib.sha256(key).digest()
    return int.from_bytes(h[:8], 'big')
