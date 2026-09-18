#!/usr/bin/env python3
"""
Communication matrices at 0.20 / 0.30 SC density.

Formulas follow the reference implementation in this package
(`legacy_count_communication_measure.py`):
  - navigation_efficiency: greedy distance-guided walk on the LENGTH matrix
  - rout_efficiency:       bct.rout_efficiency(L, transform=None)
  - search_information:    bct semantics, S (strength), transform='log'
                           (local port with numpy-2 scalar fix; bct 0.6.0's
                           retrieve_shortest_path returns a (k,1) column and
                           numpy>=2 then fancy-indexes T into arrays — the
                           port flattens the path to scalars, semantics
                           unchanged)
  - communicability:       expm(D^-1/2 S D^-1/2)
The 0.25 anchor is recomputed with the same code and compared at 6 decimal
places against the reference CSVs.
"""
import csv
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import expm
from bct import rout_efficiency, distance_wei_floyd, retrieve_shortest_path

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

BASE = PIPELINE_ROOT
MATDIR = os.path.join(BASE, 'p0_sensitivity_v1/sc_density/01_matrices')
OUTDIR = os.path.join(BASE, 'p0_sensitivity_v1/sc_density/02_comm')
RAWIN = os.path.join(BASE, 'raw_inputs')
DIST_P = os.path.join(RAWIN, 'fsaverage_parcel_distance_matrix.csv')
os.makedirs(OUTDIR, exist_ok=True)

N = 360


# ═══════════════════════════════════════ formula: navigation (length matrix)
def navigation_efficiency(L, D, max_hops=None):
    N_ = L.shape[0]
    if max_hops is None:
        max_hops = N_
    E = np.zeros((N_, N_))
    for i in range(N_):
        for j in range(N_):
            if i != j:
                curr_node = i
                last_node = curr_node
                target = j
                pl_wei = 0
                pl_bin = 0
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
                    pl_bin += 1
                    pl_wei += L[curr_node, next_node]
                    last_node = curr_node
                    curr_node = next_node
                if success and curr_node == target:
                    E[i, j] = 1 / pl_wei if pl_wei > 0 else 0
                else:
                    E[i, j] = 0
            else:
                E[i, j] = 0
    return E


# ═══════════════════════════════════════ formula: communicability (strength)
def calculate_cmy(W):
    N_ = W.shape[0]
    s = np.sum(W, axis=1)
    s_safe = np.where(s > 0, s, 1)
    D = np.diag(1 / np.sqrt(s_safe))
    W_prime = D @ W @ D
    CMY = expm(W_prime)
    return CMY


# ═══════════════════════════════════════ bct-faithful search information
def search_information_fixed(adjacency, transform=None):
    """Exact port of bct 0.6.0 search_information, with the numpy-2 fix:
    retrieve_shortest_path's (k,1) column is flattened to scalar ints."""
    N_ = len(adjacency)
    flag_triu = np.allclose(adjacency, adjacency.T)
    T = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)
    _, hops, Pmat = distance_wei_floyd(adjacency, transform)
    SI = np.zeros((N_, N_))
    SI[np.eye(N_) > 0] = np.nan
    for i in range(N_):
        for j in range(N_):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = np.asarray(retrieve_shortest_path(i, j, hops, Pmat)
                                  ).ravel().astype(np.int64)  # numpy-2 fix
                lp = len(path) - 1
                if flag_triu:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        pr_step_bk = np.zeros(lp)
                        for z in range(lp):
                            pr_step_ff[z] = T[path[z], path[z + 1]]
                            pr_step_bk[z] = T[path[z + 1], path[z]]
                        prob_sp_ff = np.prod(pr_step_ff)
                        prob_sp_bk = np.prod(pr_step_bk)
                        SI[i, j] = -np.log2(prob_sp_ff)
                        SI[j, i] = -np.log2(prob_sp_bk)
                else:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        for z in range(lp):
                            pr_step_ff[z] = T[path[z], path[z + 1]]
                        prob_sp_ff = np.prod(pr_step_ff)
                        SI[i, j] = -np.log2(prob_sp_ff)
                    else:
                        SI[i, j] = np.inf
    return SI


# ═══════════════════════════════════════ compute one density
def compute_all(L, S, D):
    t0 = time.time()
    nav = navigation_efficiency(L, D, max_hops=360)
    print(f'    nav       {time.time()-t0:7.1f}s', flush=True)
    t0 = time.time()
    rout = rout_efficiency(L, transform=None)[1]
    print(f'    rout      {time.time()-t0:7.1f}s', flush=True)
    t0 = time.time()
    si = search_information_fixed(S, transform='log')
    print(f'    search    {time.time()-t0:7.1f}s', flush=True)
    t0 = time.time()
    cm = calculate_cmy(S)
    print(f'    comm      {time.time()-t0:7.1f}s', flush=True)
    return {'nav': nav, 'rout': rout, 'search': si, 'comm': cm}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def max_abs_diff(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    nan_mismatch = int(np.isnan(a).sum() != np.isnan(b).sum())
    inf_mismatch = int(np.isinf(a).sum() != np.isinf(b).sum())
    d = np.abs(a - b)
    d = d[both]
    return (float(d.max()) if len(d) else 0.0), nan_mismatch, inf_mismatch


def main():
    which = sys.argv[1:] if len(sys.argv) > 1 else ['020', '030', 'anchor']
    D = pd.read_csv(DIST_P, index_col=0).values
    assert D.shape == (N, N)

    audit_rows = []
    for density in which:
        if density == 'anchor':
            tag = '025'
            Lpath = os.path.join(MATDIR, 'tract_length_density_025.csv')
            Spath = os.path.join(MATDIR, 'tract_strength_density_025.csv')
        else:
            tag = density
            Lpath = os.path.join(MATDIR, f'tract_length_density_{density}.csv')
            Spath = os.path.join(MATDIR, f'tract_strength_density_{density}.csv')
        L = pd.read_csv(Lpath, header=None).values.astype(np.float64)
        S = pd.read_csv(Spath, header=None).values.astype(np.float64)
        print(f'[density {density}] computing 4 matrices...', flush=True)
        mats = compute_all(L, S, D)

        if density == 'anchor':
            # compare against main-analysis 0.25 CSVs (never overwrite them)
            names = {'nav': 'navigation_efficiency_matrix',
                     'rout': 'rout_efficiency_matrix',
                     'search': 'search_information_matrix',
                     'comm': 'communicability_matrix'}
            print('[anchor validation] vs raw_inputs 0.25 CSVs:')
            val_rows = []
            for m, name in names.items():
                ref = pd.read_csv(os.path.join(
                    RAWIN, f'{name}_0.25density.csv'), header=None).values
                dmax, nanm, infm = max_abs_diff(mats[m], ref)
                print(f'  {m:8s}: max|diff|={dmax:.3e}  nan_mismatch={nanm}  '
                      f'inf_mismatch={infm}')
                val_rows.append({'metric': m, 'max_abs_diff_6dp': dmax,
                                 'nan_count_mismatch': nanm,
                                 'inf_count_mismatch': infm})
            pd.DataFrame(val_rows).to_csv(os.path.join(
                OUTDIR, 'COMM_025_ANCHOR_VALIDATION.csv'), index=False)
            continue

        names = {'nav': 'navigation_efficiency',
                 'rout': 'routing_efficiency',
                 'search': 'search_information',
                 'comm': 'communicability'}
        for m, name in names.items():
            p = os.path.join(OUTDIR, f'{name}_density_{tag}.csv')
            np.savetxt(p, mats[m], delimiter=',', fmt='%.6f')
            fin = np.isfinite(mats[m])
            vals = mats[m][fin]
            audit_rows.append({
                'density': float(tag) / 100,
                'matrix': name,
                'shape_ok': mats[m].shape == (N, N),
                'n_nonfinite': int(np.size(mats[m]) - fin.sum()),
                'diag_zero': bool(np.nanmax(np.abs(np.diag(mats[m]))) == 0)
                             if np.all(np.isfinite(np.diag(mats[m])))
                             else 'nan/inf',
                'min': float(np.nanmin(vals)) if len(vals) else np.nan,
                'max': float(np.nanmax(vals)) if len(vals) else np.nan,
                'p50': float(np.percentile(vals, 50)) if len(vals) else np.nan,
                'p99': float(np.percentile(vals, 99)) if len(vals) else np.nan,
                'sha256': sha256_file(p)[:16],
                'note': '',
            })
        print(f'[density {density}] saved 4 matrices', flush=True)

    with open(os.path.join(OUTDIR, 'COMMUNICATION_MATRIX_AUDIT.csv'), 'w',
              newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    print('audit written')


if __name__ == '__main__':
    main()
