#!/usr/bin/env python3
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cucr_engine import (CucrEngine, CucrMatrices, cell_path_stats,
                         feature_seed, K0_SAMPLE)

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, 'cucr_secondary_analysis_v1')
SRC = os.path.join(ROOT, 'fair_experiments', 'ner_per_subject')
CFG, AUD, UNI, SCR, REP = (os.path.join(OUT, d) for d in
                           ('config', 'audit', 'universe', 'screening', 'reports'))
for d in (CFG, AUD, UNI, SCR, REP):
    os.makedirs(d, exist_ok=True)

SUBPATH_P = os.path.join(SRC, 'subpath_samples_per_subject.csv')
CUCR_MATRIX_DIR = os.environ.get('CUCR_MATRIX_DIR', os.path.dirname(ROOT))
NER_PROB_P = os.path.join(CUCR_MATRIX_DIR, 'response_probability_matrix_251114.txt')
NER_ONSET_P = os.path.join(CUCR_MATRIX_DIR, 'median_onset_time_matrix_251114.txt')

# ─────────────────────────────────────────────── frozen screening constants
S10_THRESHOLD = 0.41549280879846406   # F-TRACT global median, applied verbatim
KSEL = 128                            # DDVS Ksel (registry only, pre-DDVS support)
JAC = 0.80                            # DDVS Jaccard (registry only)
K0 = 512                              # strict K0 sample
K0_R4 = 128                           # pre-declared relaxation R4
SEED_NORM = {'3_ROI': 'L_OP2-3_ROI', '46v_ROI': 'L_p9-46v_ROI'}  # NER audit rule


def log(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════
def s10_screen(eng, matrices, uni, threshold, screen_id='S10', K0v=K0,
               collect_stats=False):
    """Verbatim so_lib.s10_support: support counted PRE-DDVS, L=1 needs >=1,
    L>=2 needs >=5.  Returns (attrition_frame, stat_pool_or_none)."""
    rows, pool = [], []
    for _, r in uni.iterrows():
        si, ei = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L = int(r['path_length_edges'])
        nodes, ok, stat = cell_path_stats(eng, matrices, si, ei, L,
                                          screen_id, r['sample_id'], K0v)
        if L >= 2:
            acc = ok & (stat >= threshold)
            if collect_stats:
                pool.append(stat[ok])
        else:
            acc = ok
        n_acc = int(acc.sum())
        rows.append({'sample_id': r['sample_id'], 'L': L,
                     'seed_roi': r['seed_roi'], 'endpoint_roi': r['endpoint_roi'],
                     'n_candidate': int(nodes.shape[0]),
                     'n_b5_valid': int(ok.sum()), 'n_accepted': n_acc,
                     'in_support': bool(n_acc >= (1 if L == 1 else 5))})
    att = pd.DataFrame(rows)
    return (att, np.concatenate(pool) if pool else np.zeros(0)) \
        if collect_stats else (att, None)


def main():
    # ── engine ──────────────────────────────────────────────────────
    log('loading CUCR engine...')
    eng = CucrEngine()
    eng.load()
    matrices = CucrMatrices(eng)

    # ── 1. cell coverage from the CUCR group response matrix ────────
    prob = np.loadtxt(NER_PROB_P, dtype=np.float64)
    prob[prob < 0] = np.nan
    roi = eng.roi_order
    defined = ~np.isnan(prob)
    ypos = defined & (prob > 0)
    stim_sites = int(defined.any(axis=1).sum())
    endp_sites = int(defined.any(axis=0).sum())
    n_ypos_pairs = int(ypos.sum())
    n_yzero_pairs = int((defined & (prob == 0)).sum())

    # ── 2. observed-cell universe from the NER subpath source ───────
    log('building observed-cell universe...')
    src = pd.read_csv(SUBPATH_P)
    src['stim_region'] = src['stim_region'].replace(SEED_NORM)
    src['endpoint_region'] = src['current_end_region'].replace(SEED_NORM)
    src = src[src['stim_region'].isin(roi) & src['endpoint_region'].isin(roi)]
    src['L'] = src['subpath_length'].astype(int)
    src['triple'] = src['stim_region'] + '|' + src['endpoint_region'] + '|' \
        + src['L'].astype(str)
    triples = src.drop_duplicates('triple')
    triples = triples[['triple', 'stim_region', 'endpoint_region', 'L']].copy()
    triples['sample_id'] = 'CUCR|' + triples['stim_region'] + '|' + \
        triples['endpoint_region'] + '|' + triples['L'].astype(str)
    si = triples['stim_region'].map(eng.r2i).values
    ei = triples['endpoint_region'].map(eng.r2i).values
    triples['y_endpoint'] = prob[si, ei]
    n_all_triples = len(triples)
    n_y_nan = int(triples['y_endpoint'].isna().sum())
    n_y_zero = int((triples['y_endpoint'] == 0).sum())
    n_self = int((triples['stim_region'] == triples['endpoint_region']).sum())
    # F-TRACT mirror: universe keeps y_endpoint > 0 cells (no zero/NaN cells
    # exist in the frozen F-TRACT universe; self-pairs also absent there).
    uni = triples[triples['y_endpoint'] > 0].reset_index(drop=True)
    uni = pd.DataFrame({
        'sample_id': uni['sample_id'],
        'seed_roi': uni['stim_region'],
        'endpoint_roi': uni['endpoint_region'],
        'path_length_edges': uni['L'],
        'y_endpoint': uni['y_endpoint'],
    })
    uni.to_csv(os.path.join(UNI, 'CUCR_OBSERVED_CELL_UNIVERSE.csv'), index=False)

    # ── 3. candidate-path feasibility ────────────────────────────────
    log('computing candidate path counts...')
    Ts = []
    L_dist = {}
    for _, r in uni.iterrows():
        si_, ei_ = eng.r2i[r['seed_roi']], eng.r2i[r['endpoint_roi']]
        L_ = int(r['path_length_edges'])
        T = eng.counts(si_, ei_, L_)
        Ts.append(T)
        L_dist[L_] = L_dist.get(L_, 0) + 1
    uni['T_candidate'] = Ts
    Ts_arr = np.array(Ts)
    feas = {
        'n_cells': int(len(uni)),
        'T_median': float(np.median(Ts_arr)),
        'T_mean': float(np.mean(Ts_arr)),
        'T_min': int(Ts_arr.min()), 'T_max': int(Ts_arr.max()),
        'T_percentiles': {str(p): float(np.percentile(Ts_arr, p))
                          for p in (5, 25, 50, 75, 95)},
        'n_T_ge_512': int((Ts_arr >= 512).sum()),
        'n_T_ge_128': int((Ts_arr >= 128).sum()),
        'n_T_ge_50': int((Ts_arr >= 50).sum()),
        'n_T_ge_5': int((Ts_arr >= 5).sum()),
        'n_T_ge_1': int((Ts_arr >= 1).sum()),
        'n_T_0': int((Ts_arr == 0).sum()),
        'L_distribution': {int(k): int(v) for k, v in sorted(L_dist.items())},
    }
    with open(os.path.join(AUD, 'CANDIDATE_PATH_FEASIBILITY.json'), 'w') as f:
        json.dump(feas, f, indent=2, sort_keys=True)

    # ── 4. strict S10 screen (F-TRACT frozen rules, verbatim) ───────
    log('strict S10 screen (threshold=%.6f, K0=%d)...' % (S10_THRESHOLD, K0))
    att_strict, pool = s10_screen(eng, matrices, uni, S10_THRESHOLD,
                                  collect_stats=True)
    att_strict.to_csv(os.path.join(SCR, 'S10_EXACT_attrition.csv'), index=False)
    n_strict = int(att_strict['in_support'].sum())
    pool_median = float(np.median(pool)) if len(pool) else np.nan
    pool_p30 = float(np.percentile(pool, 30)) if len(pool) else np.nan

    # ── 5. PRE-DEFINED coverage-relaxed rules ────────────────────────
    # R1: threshold -> CUCR pool median (mirrors the F-TRACT threshold origin)
    # R2: threshold -> CUCR pool 30th percentile
    # R3: no variance threshold (B5 hard constraint only; DDVS is registry-only
    #     by the frozen convention and does not affect support)
    # R4: K0 sample 512 -> 128 (strict threshold kept)
    att_r1, _ = s10_screen(eng, matrices, uni, pool_median)
    att_r2, _ = s10_screen(eng, matrices, uni, pool_p30)
    att_r3, _ = s10_screen(eng, matrices, uni, -np.inf)
    att_r4, _ = s10_screen(eng, matrices, uni, S10_THRESHOLD, K0v=K0_R4)
    att_r1.to_csv(os.path.join(SCR, 'S10_R1_cucr_median_attrition.csv'),
                  index=False)
    att_r2.to_csv(os.path.join(SCR, 'S10_R2_cucr_p30_attrition.csv'),
                  index=False)
    att_r3.to_csv(os.path.join(SCR, 'S10_R3_b5_only_attrition.csv'),
                  index=False)
    att_r4.to_csv(os.path.join(SCR, 'S10_R4_K0128_attrition.csv'), index=False)

    relax = {
        'R1': {'rule': 'variance threshold -> CUCR pool median',
               'threshold': pool_median, 'support': int(att_r1['in_support'].sum())},
        'R2': {'rule': 'variance threshold -> CUCR pool 30th pct',
               'threshold': pool_p30, 'support': int(att_r2['in_support'].sum())},
        'R3': {'rule': 'no variance threshold (B5 only)',
               'threshold': None, 'support': int(att_r3['in_support'].sum())},
        'R4': {'rule': 'K0 sample 512 -> 128 (strict threshold)',
               'threshold': S10_THRESHOLD, 'support': int(att_r4['in_support'].sum())},
    }
    n_uni = len(uni)
    # selection rule: support rate >= 50% AND closest to strict (R1 < R2 < R3
    # in relaxation magnitude; R4 orthogonal, evaluated last)
    chosen = None
    for key in ('R1', 'R2', 'R3'):
        if relax[key]['support'] / n_uni >= 0.50:
            chosen = key
            break
    if chosen is None and relax['R4']['support'] / n_uni >= 0.50:
        chosen = 'R4'
    for key in relax:
        relax[key]['support_rate'] = relax[key]['support'] / n_uni
    strict_insufficient = (n_strict < 100) or (n_strict / n_uni < 0.30)

    # ── 6. subject grouping feasibility ──────────────────────────────
    src_l2 = src[src['L'] >= 2]
    per_sub_all = src.groupby('subject_id')['triple'].nunique()
    per_sub_l2 = src_l2.groupby('subject_id')['triple'].nunique()
    subj = {
        'n_subjects_source': int(src['subject_id'].nunique()),
        'n_subjects_l2plus': int(src_l2['subject_id'].nunique()),
        'units_per_subject_all': {
            'min': int(per_sub_all.min()), 'median': float(per_sub_all.median()),
            'max': int(per_sub_all.max())},
        'units_per_subject_l2plus': {
            'min': int(per_sub_l2.min()), 'median': float(per_sub_l2.median()),
            'max': int(per_sub_l2.max())},
        'note': 'subject identity is retained in the source data; the '
                'group-level observed-cell universe (y_endpoint from the '
                'pooled 36-subject group matrix, candidate paths from the '
                'group engine) carries no subject dimension, so the '
                'F-TRACT-frozen seed-ROI-grouped CV is the mirrored choice; '
                'subject-grouped CV is technically feasible on the 1712 '
                'per-subject units (25 subjects, 10-235 units each) but would '
                'require the single-observed-path per-subject feature design '
                '(the NER-reduced v1 design), not the candidate-path C8R.',
    }

    # ── 7. frozen config + report ────────────────────────────────────
    config = {
        'stage': 's1_sparsity',
        'screen_S10_frozen': {
            'variance_threshold_absolute': S10_THRESHOLD,
            'threshold_origin': 'F-TRACT global median, applied verbatim to '
                                'CUCR (NOT recomputed on CUCR)',
            'K0_sample': K0, 'Ksel': KSEL, 'ddvs_jaccard': JAC,
            'support_rule': 'pre-DDVS accepted count; L=1 needs >=1, '
                            'L>=2 needs >=5 (frozen convention)',
            'B5_hard': 'no repeated ROI + SC edge present + strictly temporal '
                       'onset + 4 metrics finite on every edge',
        },
        'coverage_relaxed_predeclared': {
            'R1': relax['R1']['rule'], 'R2': relax['R2']['rule'],
            'R3': relax['R3']['rule'], 'R4': relax['R4']['rule'],
            'selection_rule': 'support rate >= 50% on the observed-cell '
                             'universe, closest to strict (R1<R2<R3<R4)',
        },
        'engine': {
            'roi_order': 'roi_order_C0R2C.txt (360, same as F-TRACT)',
            'sc': 'averageConnectivity_tractLength_0.25density.csv, '
                  'mask-union symmetrized (pre-declared C0R2C mirror)',
            'latency': 'median_onset_time_matrix_251114.txt (NER pooled)',
            'metrics': '4 group-average 0.25-density matrices, bit-identical '
                       'to Task C',
        },
        'universe': {
            'definition': 'unique observed (stim, endpoint, L) cells, L>=1, '
                          'from subpath_samples_per_subject.csv; '
                          'y_endpoint > 0 from response_probability_matrix_'
                          '251114.txt',
            'n_triples_all': n_all_triples, 'n_y_nan': n_y_nan,
            'n_y_zero': n_y_zero, 'n_self_pairs': n_self,
            'n_universe': n_uni,
            'n_domain_confusion_cells_reference': 1712,
        },
        'results': {
            'matrix_coverage': {
                'n_stim_sites': stim_sites, 'n_endpoint_sites': endp_sites,
                'theoretical_pairs': stim_sites * endp_sites,
                'n_y_defined': int(defined.sum()),
                'n_y_positive': n_ypos_pairs, 'n_y_zero': n_yzero_pairs,
            },
            'candidate_paths': feas,
            'strict_s10': {
                'support': n_strict, 'universe': n_uni,
                'support_rate': n_strict / n_uni,
                'insufficient': strict_insufficient,
                'cucr_pool_median': pool_median, 'cucr_pool_p30': pool_p30,
                'n_pool_paths': int(len(pool)),
            },
            'coverage_relaxed': relax, 'chosen': chosen,
            'subjects': subj,
        },
    }
    with open(os.path.join(CFG, 'CUCR_STAGE1_CONFIG.json'), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)

    # ── report table ────────────────────────────────────────────────
    lines = []
    lines.append('# CUCR (NER) Stage 1 — Sparsity Assessment Report')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('|------|-----|')
    lines.append(f'| CUCR 刺激位点数 (rows with >=1 defined y) | {stim_sites} |')
    lines.append(f'| CUCR 终点位点数 (cols with >=1 defined y) | {endp_sites} |')
    lines.append(f'| 理论单元总数 (stim × endpoint) | {stim_sites * endp_sites} |')
    lines.append(f'| 有响应数据的单元数 (defined y; y>0) | {n_ypos_pairs} '
                 f'(+ {n_yzero_pairs} y=0 cells) |')
    lines.append('| 域混淆实验使用的 cells 数 | 1,712 '
                 '((subject, stim, endpoint, L), L>=2, 25 subjects; '
                 'UNITS.csv of ner_reduced_asymmetric_v1) |')
    lines.append(f'| 观测链 universe cells (unique (stim,endpoint,L), y>0) | '
                 f'{n_uni} |')
    lines.append(f'| 候选路径数中位数 (T at cell L) | {feas["T_median"]:.1f} |')
    lines.append(f'| 候选路径数 ≥ 512 的单元数 | {feas["n_T_ge_512"]} |')
    lines.append(f'| 候选路径数 ≥ 128 的单元数 | {feas["n_T_ge_128"]} |')
    lines.append(f'| 候选路径数 ≥ 5 的单元数 | {feas["n_T_ge_5"]} |')
    lines.append(f'| 候选路径数 = 0 的单元数 | {feas["n_T_0"]} |')
    lines.append(f'| 严格 S10 筛选后支持数 | {n_strict} |')
    lines.append(f'| 严格 S10 支持率 | {n_strict / n_uni:.1%} '
                 f'({"INSUFFICIENT" if strict_insufficient else "sufficient"}) |')
    chosen_txt = chosen if chosen else 'NONE meets >=50%'
    if chosen:
        lines.append(f'| Coverage-relaxed 方案 | {chosen} — '
                     f'{relax[chosen]["rule"]} |')
        lines.append(f'| Coverage-relaxed 支持数 | {relax[chosen]["support"]} |')
        lines.append(f'| Coverage-relaxed 支持率 | '
                     f'{relax[chosen]["support_rate"]:.1%} |')
    else:
        lines.append('| Coverage-relaxed 方案 | none of R1-R4 reaches 50% |')
    lines.append(f'| 受试者身份是否保留 | 是（源数据 {subj["n_subjects_source"]} 名受试者；'
                 f'L≥2 单元 {subj["n_subjects_l2plus"]} 名） |')
    lines.append('| 推荐 CV 分组方式 | stimulation-ROI-grouped '
                 '(F-TRACT frozen mirror；subject-grouped 在 per-subject 单元设计下可行，'
                 '但不适用于 group-level 候选路径 universe，理由见 6) |')
    lines.append('')
    lines.append('## Relaxed-rule support (pre-declared)')
    lines.append('| 方案 | 规则 | 阈值 | 支持数 | 支持率 |')
    lines.append('|------|------|------|--------|--------|')
    for k, v in relax.items():
        lines.append(f'| {k} | {v["rule"]} | '
                     f'{v["threshold"] if v["threshold"] is not None else "—"} | '
                     f'{v["support"]} | {v["support_rate"]:.1%} |')
    lines.append('')
    lines.append(f'CUCR candidate-path pool: {len(pool)} B5-valid L>=2 paths; '
                 f'median {pool_median:.4f}, 30th pct {pool_p30:.4f} '
                 f'(F-TRACT frozen threshold 0.4155 for comparison).')
    with open(os.path.join(REP, 'CUCR_SPARSITY_REPORT.md'), 'w') as f:
        f.write('\n'.join(lines))
    log('wrote report + config + attrition CSVs')
    log('strict support=%d/%d (%.1f%%) chosen relaxed=%s'
        % (n_strict, n_uni, 100.0 * n_strict / n_uni, chosen))


if __name__ == '__main__':
    main()
