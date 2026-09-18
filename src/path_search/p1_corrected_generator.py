"""P1 — Corrected time-ordered SC candidate chain generator."""
import numpy as np
from collections import defaultdict

def generate_p1_paths(stim_idx, sc_matrix, latency_row, region_names, hemisphere_indices=None):
    """Corrected: full-precision latency, DAG construction, tuple-based dedup,
    no level replacement, within-hemisphere scope matching legacy."""
    if hemisphere_indices is None:
        hemisphere_indices = list(range(len(latency_row)))
    valid_nodes = [(stim_idx, 0.0)]
    for resp_idx in hemisphere_indices:
        if resp_idx == stim_idx: continue
        tv = latency_row[resp_idx]
        if np.isnan(tv) or tv <= 0: continue
        valid_nodes.append((resp_idx, float(tv)))
    if len(valid_nodes) <= 1: return []
    edges = defaultdict(list)
    for ui, ut in valid_nodes:
        for vi, vt in valid_nodes:
            if ut >= vt: continue
            if sc_matrix[ui, vi] > 0:
                edges[ui].append(vi)
    all_paths = set()
    def dfs(current, path_tuple):
        path_set = set(path_tuple)
        for nb in edges.get(current, []):
            if nb in path_set: continue
            new_path = path_tuple + (nb,)
            all_paths.add(new_path)
            dfs(nb, new_path)
    dfs(stim_idx, (stim_idx,))
    return [[int(x) for x in p] for p in all_paths if len(p) >= 2]
