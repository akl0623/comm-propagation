#!/usr/bin/env python3
import os
import sys
import json
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

R = os.environ.get('PIPELINE_ROOT', '/mnt/d/revision_v2')
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.join(PACKAGE_ROOT, 'pipeline')
sys.path[:0] = [
    os.path.join(CODE_ROOT, 's10_optimized_v1', 'code'),
    os.path.join(CODE_ROOT, 's10_pathlevel_v1', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1', 'confirmatory', 'code'),
    os.path.join(CODE_ROOT, 'screen_match_confirm_v1', 'candidate_screens', 'code'),
    os.path.join(CODE_ROOT, 'iterative_combined_screen_v1', 'full_experiment', 'code'),
]

INFERENCE_DIR = os.path.join(PACKAGE_ROOT, 'src', 'inference')
sys.path.insert(0, INFERENCE_DIR)
from portable_inference import prepare, inference

import bridge_noscreen as B                      # noqa: E402
import stage4a_lib as A4                         # noqa: E402
import pl_lib as PL                              # noqa: E402
import so_lib as SO                              # noqa: E402

# ── re-point every absolute path from the original machine to this mirror ──
SM = os.path.join(R, 'screen_match_confirm_v1')
BI = os.path.join(R, 'frozen_s0', 'bridge_inputs')
RAW = os.path.join(R, 'raw_inputs')
OUT = os.path.join(R, 's10_optimized_v1')

B.ROOT = R
B.SMCR = SM
B.RAW = RAW
B.FS0 = os.path.join(R, 'frozen_s0')
B.BI = BI
B.LAT_P = os.path.join(RAW, 'Reordered_matrix_onset_delay__median.csv')
B.NAV_P = os.path.join(RAW, 'navigation_efficiency_matrix_0.25density.csv')
B.ROUT_P = os.path.join(RAW, 'rout_efficiency_matrix_0.25density.csv')
B.SRCH_P = os.path.join(RAW, 'search_information_matrix_0.25density.csv')
B.COMM_P = os.path.join(RAW, 'communicability_matrix_0.25density.csv')
B.SC_P = os.path.join(
    BI, 'averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv')
B.ROI_P = os.path.join(BI, 'roi_order_C0R2C.txt')
B.UNI_P = os.path.join(BI, 'taskC_primary_2636_universe_C0R2D.csv')
B.FOLD_P = os.path.join(BI, 'taskC_primary_fold_manifest_C0R2D_v2.csv')
A4.SMCR = SM
A4.RAW = B.RAW
A4.BI = B.BI
A4.UNI_P = B.UNI_P
A4.FOLD_P = B.FOLD_P
PL.ROOT = R
PL.SMCR = SM
PL.ICS = os.path.join(R, 'iterative_combined_screen_v1')
PL.OUT = os.path.join(R, 's10_pathlevel_v1')
PL.SMK = os.path.join(PL.ICS, 'smoke_test', 'ROUND2')
PL.REGISTRY_P = os.path.join(PL.SMK, 'S10_registry.parquet')
PL.ATTRITION_P = os.path.join(PL.SMK, 'S10_attrition.csv')
PL.PRIOR_TASKC = os.path.join(PL.ICS, 'full_experiment', 'S10', 'taskC')
SO.ROOT = R
SO.SMCR = SM
SO.ICS = PL.ICS
SO.B4R = os.path.join(R, 'b4_confirmatory_v1')
SO.PLV = PL.OUT
SO.OUT = OUT
SO.SENS = os.path.join(OUT, 'sensitivity')
SO.PSETS = os.path.join(OUT, 'strata', '_path_sets')

RESULT_DIR = os.path.join(R, 'perm_review_v1', 'results')
RUN_DIR = os.path.join(R, 'perm_review_v1', 'threshold_fullcv')
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

N_WORKERS = int(os.environ.get('N_WORKERS', 8))
ARMS = ['C4', 'C8R-All']
N_PERM = 10000

G = {}


def log(m):
    print(m, flush=True)


def _fold_task(args):
    arm, rep, fold = args
    ctx = G['ctx']
    tr, te, test_sids, groups = SO.fold_ids(ctx, rep, fold)
    return SO.run_fold(G['X'][arm], ctx['y'], tr, te, groups, test_sids,
                       rep, fold, arm)


def build_ctx(tag, uni_full, fm):
    att = pd.read_csv(os.path.join(SO.SENS, f'SUPPORT_{tag}.csv'))
    ctx = SO.context_from_support(uni_full, att, 'S10', 'y_endpoint', fm, tag)
    return ctx, att


def run_tag(tag, uni_full, fm):
    t0 = time.time()
    ctx, att = build_ctx(tag, uni_full, fm)
    # T50 was never run as a smoke, so it has no cached path set of its own;
    # its support is identical to H1 and the paths are sampled with the same
    # screen id ('S10') and cell seeds, so H1's cached set is the same set.
    ps_tag = 'H1' if tag == 'T50' else tag
    pset = SO._read_pset(os.path.join(SO.PSETS, f'{ps_tag}__all_sampled.parquet'))
    c4 = SO.base_c4(ctx)
    x8, n_paths = SO.build_c8r(ctx, pset, c4)
    G['ctx'] = ctx
    G['X'] = {'C4': c4, 'C8R-All': x8}
    log(f'[{tag}] support {len(ctx["uni"])} cells, features '
        f'C4 {c4.shape} C8R {x8.shape}, median paths/cell '
        f'{np.median(n_paths):.0f} ({time.time()-t0:.0f}s)')

    reps = sorted(ctx['fm_sup']['repeat'].unique())
    folds = sorted(ctx['fm_sup']['test_fold'].unique())
    tasks = [(a, r, f) for a in ARMS for r in reps for f in folds]
    t1 = time.time()
    rows, fms = [], []
    with mp.Pool(N_WORKERS) as pool:
        for oof, fm_ in pool.imap_unordered(_fold_task, tasks):
            rows.extend(oof)
            fms.append(fm_)
    oof = pd.DataFrame(rows).sort_values(
        ['arm', 'repeat', 'outer_fold', 'sample_id']).reset_index(drop=True)
    foldm = pd.DataFrame(fms)
    log(f'[{tag}] CV done: {len(tasks)} folds in {time.time()-t1:.0f}s '
        f'({len(reps)} repeats x {len(folds)} folds x {len(ARMS)} arms)')

    a = oof[oof['arm'] == 'C8R-All'].drop(columns=['arm'])
    b = oof[oof['arm'] == 'C4'].drop(columns=['arm'])
    r2 = {arm: SO.repeat_r2(oof[oof['arm'] == arm]) for arm in ARMS}
    obs = r2['C8R-All'] - r2['C4']
    smap = ctx['uni'][['sample_id', 'seed_roi']].drop_duplicates('sample_id')

    t2 = time.time()
    lo, hi, frac, nboot = SO.paired_bootstrap(a, b, smap)
    log(f'[{tag}] bootstrap CI [{lo:+.4f}, {hi:+.4f}] ({time.time()-t2:.0f}s)')

    t3 = time.time()
    diagnostic = inference(prepare(a, b, smap), 'cluster', N_PERM,
                                        f'threshold_fullcv|{tag}')
    obs_p, p, p1 = (diagnostic['delta_r2'], diagnostic['p_two_sided'],
                    diagnostic['p_two_sided_plus1'])
    assert abs(obs_p - obs) < 1e-12
    n_extreme = diagnostic['upper_tail_count']
    log(f'[{tag}] permutation p = {p:.4f} (10,000 draws, cluster-linked; '
        f'{n_extreme} exceed) ({time.time()-t3:.0f}s)')

    oof.to_parquet(os.path.join(RUN_DIR, f'OOF_FULLCV_{tag}.parquet'),
                   index=False)
    foldm.to_csv(os.path.join(RUN_DIR, f'FOLDS_FULLCV_{tag}.csv'), index=False)
    row = {
        'threshold': tag,
        'n_support': int(len(ctx['uni'])),
        'support_pct': 100.0 * len(ctx['uni']) / 2636,
        'c4_r2': r2['C4'], 'c8r_r2': r2['C8R-All'], 'delta_r2': obs,
        'ci_lo': lo, 'ci_hi': hi, 'boot_frac_gt_0': frac, 'n_boot': nboot,
        'p_perm': p, 'p_perm_plus1': p1,
        'n_exceed': n_extreme,
        'n_repeats_positive': int(sum(
            1 for r in reps
            if SO.repeat_r2(a[a['repeat'] == r]) > SO.repeat_r2(b[b['repeat'] == r]))),
        'n_repeats': len(reps),
        'protocol': '5 repeats x 5 folds (full)',
        'wall_s': round(time.time() - t0, 1),
    }
    log(f'[{tag}] R2 C4={row["c4_r2"]:.4f} C8R={row["c8r_r2"]:.4f} '
        f'dR2={obs:+.4f} p={p:.4f} repeats+ {row["n_repeats_positive"]}/{len(reps)}')
    return row


def main():
    tags = sys.argv[1:] or ['T50', 'T30', 'T40', 'T60', 'T70']
    log('=' * 70)
    log('Threshold sensitivity at the FULL protocol (5 x 5)')
    log('=' * 70)
    uni_full = pd.read_csv(B.UNI_P)
    fm = pd.read_csv(B.FOLD_P)
    eng = SO.engine()
    SO.build_matrices(eng)
    log(f'engine ready (workers={N_WORKERS})')

    out_rows = []
    for tag in tags:
        row = run_tag(tag, uni_full, fm)
        out_rows.append(row)
        pd.DataFrame(out_rows).to_csv(
            os.path.join(RESULT_DIR, 'THRESHOLD_FULLCV_RESULTS.csv'),
            index=False)

        if tag == 'T50':
            ok = (abs(row['c4_r2'] - 0.436874980042614) < 1e-9
                  and abs(row['c8r_r2'] - 0.4580163658477989) < 1e-9)
            log(f'[VALIDATION] T50 reproduces published H1 numbers: {ok}')
            if not ok:
                raise SystemExit('T50 does not reproduce H1 — aborting.')
    log(f"done -> {os.path.join(RESULT_DIR, 'THRESHOLD_FULLCV_RESULTS.csv')}")


if __name__ == '__main__':
    main()
