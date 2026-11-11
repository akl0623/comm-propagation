#!/usr/bin/env python3
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

SMCR = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/screen_match_confirm_v1'
PLV = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/s10_pathlevel_v1'
SOV = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/s10_optimized_v1'
ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
for p in (os.path.join(SOV, 'code'),
          os.path.join(SMCR, 'confirmatory', 'code'),
          os.path.join(SMCR, 'candidate_screens', 'code'),
          os.path.join(PLV, 'code')):
    sys.path.insert(0, p)

import bridge_noscreen as B                                          # noqa: E402
import stage4a_lib as A4                                             # noqa: E402
import so_lib as SO                                                  # noqa: E402

P0 = os.path.join(ROOT, 'p0_sensitivity_v1/bin_width')
PATHS = os.path.join(P0, '01_paths')
AUDIT = os.path.join(P0, '02_audit')
MODEL = os.path.join(P0, '03_model')
for d in (PATHS, AUDIT, MODEL):
    os.makedirs(d, exist_ok=True)

CONDITIONS = {'bin_1ms': {'W': 1, 'quantise': False,
                          'tag': 'H1|C8R_All_minus_C4|bin1ms'},
              'bin_2ms': {'W': 2, 'quantise': True,
                          'tag': 'H1|C8R_All_minus_C4|bin2ms'},
              'bin_5ms': {'W': 5, 'quantise': True,
                          'tag': 'H1|C8R_All_minus_C4|bin5ms'}}
THRESHOLD = SO.S10_THRESHOLD
K0 = SO.K0_SAMPLE
N_WORKERS = 14
G = {}


# ══════════════════════════════════════════════════════════════════ latency
RAW_LAT = os.path.join(ROOT, 'raw_inputs',
                       'Reordered_matrix_onset_delay__median.csv')


def make_binned_latency(W):
    # ALWAYS quantise the RAW latencies (B.LAT_P is mutated per condition)
    lat = pd.read_csv(RAW_LAT, index_col=0)
    V = lat.values.astype(float)
    Vb = V.copy()
    fin = np.isfinite(V)
    Vb[fin] = np.floor(V[fin] / W + 0.5) * W   # nearest boundary, halves up
    out = os.path.join(PATHS, f'onset_bin_{W}ms.csv')
    pd.DataFrame(Vb, index=lat.index, columns=lat.columns).to_csv(out)
    n_bin0 = int(((Vb == 0) & fin & (V > 0)).sum())
    return out, n_bin0


# ══════════════════════════════════════════════════════════════════ path search
def search_cell(args):
    """One cell's K0 path set — EXACT replication of so_lib.build_all_sampled."""
    si, ei, L, sid, screen_id = args
    eng = G['eng']
    ranks = eng.sample_ranks(si, ei, L, K0, SO.feature_seed(screen_id, sid))
    arr = eng.unrank_cell(si, ei, L, ranks)
    return sid, [[int(x) for x in row] for row in arr]


def build_all_sampled_parallel(ctx):
    tasks = []
    for _, r in ctx['uni'].iterrows():
        si, ei = ctx['eng'].r2i[r['seed_roi']], ctx['eng'].r2i[r['endpoint_roi']]
        tasks.append((si, ei, int(r['path_length_edges']), r['sample_id'],
                      ctx['screen_id']))
    out = {}
    with mp.Pool(N_WORKERS) as pool:
        for sid, paths in pool.imap_unordered(search_cell, tasks):
            out[sid] = paths
    return out


# ══════════════════════════════════════════════════════════════════ screen
def screen_cell(eng, matrices, nodes, L, si):
    """so_lib.cell_path_stats math on an explicit node array (no re-search)."""
    if not nodes:
        return np.zeros(0, bool), np.zeros(0)
    arr = np.asarray(nodes, dtype=np.int64)
    u, v = arr[:, :-1], arr[:, 1:]
    srt = np.sort(arr, axis=1)
    ok = ~(np.diff(srt, axis=1) == 0).any(axis=1)
    ok &= (matrices.sc_mask[u, v] == 1).all(axis=1)
    vals = eng.lat[si][arr].astype(np.float64).copy()
    vals[:, 0] = 0.0
    ok &= (np.diff(vals, axis=1) > 0).all(axis=1)
    for m in SO.METRICS:
        ok &= np.isfinite(getattr(matrices, m)[u, v]).all(axis=1)
    if L < 2:
        return ok, np.full(len(arr), np.nan)
    stat = np.zeros(len(arr), dtype=np.float64)
    for m in SO.METRICS:
        stat += np.var(getattr(matrices, m)[u, v], axis=1, ddof=0) / 4.0
    return ok, stat


def apply_screen(ctx, matrices, all_paths):
    rows = []
    for _, r in ctx['uni'].iterrows():
        L = int(r['path_length_edges'])
        sid = r['sample_id']
        si = ctx['eng'].r2i[r['seed_roi']]
        nodes = all_paths.get(sid, [])
        ok, stat = screen_cell(ctx['eng'], matrices, nodes, L, si)
        if L >= 2:
            acc = ok & (stat >= THRESHOLD)
        else:
            acc = ok
        n_acc = int(acc.sum())
        rows.append({'sample_id': sid, 'L': L, 'seed_roi': r['seed_roi'],
                     'endpoint_roi': r['endpoint_roi'],
                     'n_candidate': int(len(nodes)),
                     'n_b5_valid': int(ok.sum()), 'n_accepted': n_acc,
                     'in_support': bool(n_acc >= (1 if L == 1 else 5))})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════ audit
def audit_vs_anchor(bin_key, all_paths, anchor_paths, att):
    cells_self = {s: p for s, p in all_paths.items() if p}
    cells_anchor = {s: p for s, p in anchor_paths.items() if p}
    common = set(cells_self) & set(cells_anchor)
    jac, topk = [], []
    for s in common:
        es = {tuple(x) for x in cells_self[s]}
        ea = {tuple(x) for x in cells_anchor[s]}
        jac.append(len(es & ea) / max(1, len(es | ea)))
        topk.append(len(es & ea) / max(1, len(ea)))
    med_L = float(np.median([len(n) - 1 for pl in cells_self.values()
                             for n in pl])) if cells_self else np.nan
    return {
        'n_cells_with_paths': len(cells_self),
        'mean_paths_per_cell': float(np.mean([len(p) for p in cells_self.values()]))
                               if cells_self else np.nan,
        'median_path_length': med_L,
        'attrition_rate': float(1 - att['in_support'].astype(bool).mean()),
        'n_support': int(att['in_support'].astype(bool).sum()),
        'edge_jaccard_vs_1ms': float(np.mean(jac)) if jac else np.nan,
        'topk_overlap_vs_1ms': float(np.mean(topk)) if topk else np.nan,
        'n_common_cells': len(common),
    }


# ══════════════════════════════════════════════════════════════════ model
def _task(args):
    arm, rep, fold = args
    ctx = G['ctx']
    tr, te, test_sids, groups = SO.fold_ids(ctx, rep, fold)
    return SO.run_fold(G['X'][arm], ctx['y'], tr, te, groups, test_sids,
                       rep, fold, arm)


def run_models(ctx_bin, path_lists, cond_key, cond):
    t0 = time.time()
    c4 = SO.base_c4(ctx_bin)
    X_c8r, n_paths = SO.build_c8r(ctx_bin, path_lists, c4)
    G['ctx'] = ctx_bin
    G['X'] = {'C4': c4, 'C8R-All': X_c8r}
    print(f'  features: C4 {c4.shape}, C8R {X_c8r.shape}, '
          f'mean paths/cell {n_paths.mean():.1f} ({time.time()-t0:.0f}s)',
          flush=True)
    outdir = os.path.join(MODEL, cond_key)
    os.makedirs(outdir, exist_ok=True)
    oof_parts, overall = {}, []
    for arm in ('C4', 'C8R-All'):
        tasks = [(arm, rep, fold) for rep in range(5) for fold in range(5)]
        oof_rows = []
        with mp.Pool(N_WORKERS) as pool:
            for oof, _fm in pool.imap_unordered(_task, tasks):
                oof_rows.extend(oof)
        odf = pd.DataFrame(oof_rows)
        name = 'C4' if arm == 'C4' else 'C8R'
        odf.to_parquet(os.path.join(outdir, f'OOF_{name}.parquet'), index=False)
        oof_parts[arm] = odf
        rep_r2 = [A4.metric_block(g['y_true'].values, g['y_pred'].values)['r2']
                  for _, g in odf.groupby('repeat')]
        rep_r2 = [v for v in rep_r2 if np.isfinite(v)]
        overall.append({'stratum': cond_key, 'arm': arm,
                        'level': 'repeat_summary', 'r2': float(np.mean(rep_r2)),
                        'n_repeats_used': len(rep_r2),
                        'per_repeat_r2': json.dumps(rep_r2)})
        print(f'  {arm}: 25 folds done ({time.time()-t0:.0f}s)', flush=True)
    pd.DataFrame(overall).to_csv(os.path.join(outdir, 'OVERALL_METRICS.csv'),
                                 index=False)
    smap = SO.seed_map(ctx_bin)
    row = SO.contrast(oof_parts['C8R-All'], oof_parts['C4'], smap,
                      'C8R_All_minus_C4', do_perm=True, tag=cond['tag'])
    row['condition'] = cond_key
    print(f'  contrast: dR2={row["point_estimate"]:+.6f} '
          f'CI[{row["bootstrap_ci_lo"]:+.4f},{row["bootstrap_ci_hi"]:+.4f}] '
          f'p={row["permutation_p_two_sided"]:.4f} '
          f'repeats+ {row["n_repeats_positive"]}/{row["n_repeats"]} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return row, oof_parts


# ══════════════════════════════════════════════════════════════════ driver
def run_condition(cond_key, cond, anchor_paths=None, anchor_att=None):
    t0 = time.time()
    print(f'\n=== {cond_key} ===', flush=True)
    if cond['quantise']:
        lat_csv, n_bin0 = make_binned_latency(cond['W'])
        B.LAT_P = lat_csv
        print(f'  binned latency -> {lat_csv} (n pairs in bin 0: {n_bin0})',
              flush=True)
    else:
        B.LAT_P = RAW_LAT
    A4.ENGINE = None
    SO._ENG = None
    B.ENGINE = None
    ctx = SO.load_context('H1')
    eng = ctx['eng']
    assert eng is B.ENGINE
    G['eng'] = eng

    # path search (K0=512, frozen seeds)
    all_paths = build_all_sampled_parallel(ctx)
    n_tot = sum(len(p) for p in all_paths.values())
    print(f'  paths: {n_tot} across {len(all_paths)} cells '
          f'({time.time()-t0:.0f}s)', flush=True)

    # S10 screen on the new path sets
    matrices = SO.build_matrices(eng)
    att = apply_screen(ctx, matrices, all_paths)
    print(f'  support: {int(att["in_support"].sum())}/1582 '
          f'({time.time()-t0:.0f}s)', flush=True)

    # save path library (cells with >=1 path)
    lib_dir = os.path.join(PATHS, cond_key)
    os.makedirs(lib_dir, exist_ok=True)
    rows = [{'sample_id': s, 'path_index': k, 'L': len(n) - 1,
             'roi_id_sequence': '|'.join(str(int(x)) for x in n)}
            for s, pl in all_paths.items() for k, n in enumerate(pl)]
    pd.DataFrame(rows).to_parquet(
        os.path.join(lib_dir, 'path_library.parquet'), index=False)
    att.to_csv(os.path.join(AUDIT, f'{cond_key}_ATTRITION.csv'), index=False)

    # anchor validation
    if cond_key == 'bin_1ms':
        ref = SO.load_path_set('H1', 'all_sampled')
        same = (set(ref) == set(all_paths) and
                all(ref[s] == all_paths[s] for s in ref))
        print(f'  ANCHOR CHECK: path library bit-identical to H1__all_sampled: '
              f'{same}', flush=True)
        frozen = pd.read_csv(os.path.join(
            ROOT, 'iterative_combined_screen_v1/smoke_test/ROUND2',
            'S10_attrition.csv'))
        m = att.merge(frozen[['sample_id', 'in_support']], on='sample_id',
                      suffixes=('_new', '_frozen'))
        support_match = bool((m['in_support_new'].astype(bool) ==
                              m['in_support_frozen'].astype(bool)).all())
        print(f'  ANCHOR CHECK: support matches frozen S10 attrition: '
              f'{support_match}', flush=True)

    # audit vs 1ms anchor
    aud = {'bin_width': cond['W'], **audit_vs_anchor(
        cond_key, all_paths, anchor_paths or all_paths, att)}
    if cond_key == 'bin_1ms':
        aud.update({'edge_jaccard_vs_1ms': 1.0, 'topk_overlap_vs_1ms': 1.0})

    # models on the surviving support
    uni_full = pd.read_csv(B.UNI_P)
    fm = pd.read_csv(B.FOLD_P)
    ctx_bin = SO.context_from_support(uni_full, att, 'S10', 'y_endpoint', fm,
                                      cond_key)
    pl_sup = {s: p for s, p in all_paths.items()
              if s in ctx_bin['support_sids']}
    row, oof_parts = run_models(ctx_bin, pl_sup, cond_key, cond)
    return row, oof_parts, att, aud


def main():
    targets = sys.argv[1:] or ['bin_1ms', 'bin_2ms', 'bin_5ms']
    anchor_paths = anchor_att = None
    rows, audits = [], []
    for key in targets:
        cond = CONDITIONS[key]
        row, oof_parts, att, aud = run_condition(
            key, cond, anchor_paths=anchor_paths, anchor_att=anchor_att)
        rows.append(row)
        audits.append(aud)
        if key == 'bin_1ms':
            anchor_paths = {s: p for s, p in
                            SO.load_path_set('H1', 'all_sampled').items()}
    pd.DataFrame(audits).to_csv(os.path.join(AUDIT, 'PATH_SET_AUDIT.csv'),
                                index=False)
    cols = ['condition', 'r2_a', 'r2_b', 'point_estimate', 'bootstrap_ci_lo',
            'bootstrap_ci_hi', 'bootstrap_frac_gt_0', 'n_boot_finite',
            'n_repeats_positive', 'n_repeats', 'repeats',
            'permutation_p_two_sided', 'perm_obs']
    pd.DataFrame(rows)[cols].to_csv(os.path.join(P0, 'ALL_BINS_CONTRAST.csv'),
                                    index=False)
    print('\n=== ALL BINS ===', flush=True)
    for r in rows:
        print(f'  {r["condition"]:8s} dR2={r["point_estimate"]:+.6f}  '
              f'CI[{r["bootstrap_ci_lo"]:+.4f},{r["bootstrap_ci_hi"]:+.4f}]  '
              f'p={r["permutation_p_two_sided"]:.4f}  '
              f'repeats+ {r["n_repeats_positive"]}/{r["n_repeats"]}',
              flush=True)
    print('\n=== PATH SET AUDIT ===', flush=True)
    for a in audits:
        print(f'  {a["bin_width"]}ms: cells_with_paths={a["n_cells_with_paths"]} '
              f'mean_paths={a["mean_paths_per_cell"]:.1f} '
              f'attrition={a["attrition_rate"]:.3f} '
              f'edge_jac={a["edge_jaccard_vs_1ms"]:.3f} '
              f'topk={a["topk_overlap_vs_1ms"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
