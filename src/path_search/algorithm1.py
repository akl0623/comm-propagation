"""P0 — Legacy exact path generator. Reproduces find pathway.py behavior."""
import numpy as np
from collections import defaultdict

def generate_p0_paths(stim_idx, sc_matrix, latency_row, region_names, hemisphere_indices=None):
    """Exact reproduction: integer-rounded time bins, paths=new_paths replacement,
    string-based containment pruning, same-hemisphere scope."""
    if hemisphere_indices is None:
        hemisphere_indices = list(range(len(latency_row)))
    nodes = [(stim_idx, 0.0)]
    for resp_idx in hemisphere_indices:
        if resp_idx == stim_idx: continue
        tv = latency_row[resp_idx]
        if np.isnan(tv) or tv <= 0: continue
        nodes.append((resp_idx, float(tv)))
    if len(nodes) <= 1: return []
    nodes.sort(key=lambda x: x[1])
    time_levels = {}
    for idx, t in nodes:
        if t == 0.0: continue
        tk = round(t)
        if tk not in time_levels: time_levels[tk] = []
        time_levels[tk].append(idx)
    sorted_levels = sorted(time_levels.items())
    paths = [([stim_idx], 0)]
    all_paths = []
    for time_key, level_nodes in sorted_levels:
        new_paths = []
        for path_nodes, last_time in paths:
            valid = [n for n in level_nodes if latency_row[n] > last_time]
            for node in valid:
                last_node = path_nodes[-1]
                if sc_matrix[last_node, node] <= 0: continue
                new_paths.append((path_nodes + [node], latency_row[node]))
        all_paths.extend([p for p, _ in new_paths])
        paths = new_paths
    all_paths.append([stim_idx])
    all_paths.sort(key=len, reverse=True)
    final = []
    for p in all_paths:
        if len(p) < 2: continue
        p_str = '→'.join([region_names[i] for i in p])
        contained = any(p_str in '→'.join([region_names[i] for i in e]) for e in final)
        if not contained: final.append(p)
    return final
