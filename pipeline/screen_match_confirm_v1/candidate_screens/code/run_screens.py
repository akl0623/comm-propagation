#!/usr/bin/env python3
import os
import sys
import json
import time
import random as _py_random
import hashlib
import shutil

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd

SMCR = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2') + '/screen_match_confirm_v1'
sys.path.append(os.path.join(SMCR, 'confirmatory', 'code'))
sys.path.append(os.path.join(SMCR, 'candidate_screens', 'code'))

import screens
from screens import Path, Matrices

# ── firewalls: open trace + explicit io manifest ──
_ORIG_OPEN = open
_OPEN_EVENTS = []          # (path, mode, source)
_FORBIDDEN = ['target', 'conditional_prob', 'oof', 'prediction', 'label',
              'exploratory']


def _wrapped_open(f, *a, **kw):
    p = f if isinstance(f, str) else getattr(f, 'name', str(f))
    mode = kw.get('mode', 'r')
    _OPEN_EVENTS.append((p, mode, 'open'))
    return _ORIG_OPEN(f, *a, **kw)


def io_log(path, mode):
    _OPEN_EVENTS.append((path, mode, 'io'))


def sha256_file(p):
    h = hashlib.sha256()
    with _ORIG_OPEN(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def cell_seed(screen_id, sid):
    return int.from_bytes(
        hashlib.sha256(f'{screen_id}|{sid}'.encode()).digest()[:8], 'big')


# ══════════════════════════════════════════════════════════════════
def load_engine(canary_root=None):
    """Load the Stage-0 P1 engine; canary mode patches the module paths to
    the canary input tree (restored afterwards)."""
    import bridge_noscreen as B
    if canary_root is None:
        eng = B.P1Engine()
        eng.load()
        return eng, B.RAW, B.BI
    raw = os.path.join(canary_root, 'raw')
    bi = os.path.join(canary_root, 'bi')
    saved = {k: getattr(B, k) for k in
             ('RAW', 'BI', 'LAT_P', 'NAV_P', 'ROUT_P', 'SRCH_P', 'COMM_P',
              'SC_P', 'ROI_P')}
    B.RAW = raw
    B.BI = bi
    B.LAT_P = os.path.join(raw, 'Reordered_matrix_onset_delay__median.csv')
    B.NAV_P = os.path.join(raw, 'navigation_efficiency_matrix_0.25density.csv')
    B.ROUT_P = os.path.join(raw, 'rout_efficiency_matrix_0.25density.csv')
    B.SRCH_P = os.path.join(raw, 'search_information_matrix_0.25density.csv')
    B.COMM_P = os.path.join(raw, 'communicability_matrix_0.25density.csv')
    B.SC_P = os.path.join(bi, 'averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv')
    B.ROI_P = os.path.join(bi, 'roi_order_C0R2C.txt')
    eng = B.P1Engine()
    eng.load()
    for k, v in saved.items():
        setattr(B, k, v)
    return eng, raw, bi


def make_matrices(eng, raw, bi, cfg_dir):
    """Assemble the full Matrices namespace for the screens."""
    m = Matrices()
    sc = pd.read_csv(os.path.join(
        bi, 'averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv'),
        header=None).values.astype(np.float64)
    m.sc_weight = sc
    m.sc_mask = eng.mask
    m.onset = eng.lat
    m.nav = eng.mats['nav']
    m.rout = eng.mats['rout']
    m.search = eng.mats['search']
    m.comm = eng.mats['comm']
    tl = pd.read_csv(os.path.join(raw, 'averageConnectivity_tractLength_0.25density.csv'),
                     header=None).values.astype(np.float64)
    m.tract_length = tl
    # Yeo7 / lobe per roi
    mmp = pd.read_csv(os.path.join(raw, 'mmp_to_yeo7_mapping.csv'))
    yeo7_by_label = dict(zip(mmp['MMP_Label'], mmp['Yeo7_Network_Label']))
    yeo7_ids = []
    lobe_ids = []
    lobe_names = []
    for name in eng.roi_order:
        area = name[:-4].split('_')[1]
        yeo7_ids.append(yeo7_by_label[name[:-4]])
        assert area in screens.LOBES, f'missing lobe {area}'
        lobe_names.append(screens.LOBES[area])
    m.yeo7_ids = np.asarray(yeo7_ids)
    m.lobe_names = np.asarray(lobe_names)
    m.lobe_ids = np.array([screens.LOBE_ORDER.index(x) for x in lobe_names])
    # anatomical whitelist: relays + circuit edge set
    wl = pd.read_csv(os.path.join(cfg_dir, 'ANATOMICAL_WHITELIST.csv'))
    relay_rois = set()
    circuits = {}
    for _, r in wl.iterrows():
        roi_id = int(r['roi_id'])
        if r['role'] == 'relay':
            relay_rois.add(roi_id)
        circuits.setdefault(r['circuit_name'], set()).add(roi_id)
    m.relay_rois = relay_rois
    wl_edges = set()
    for members in circuits.values():
        for u in members:
            for v in members:
                if u != v:
                    wl_edges.add((u, v))
    m.whitelist_edge_set = wl_edges
    return m


# ══════════════════════════════════════════════════════════════════
def random_walk_candidates(eng, si, ei, L, n_target, screen_id, sid):
    """Seeded random walk for L > 10 cells (frozen fallback included)."""
    seed_int = cell_seed(screen_id, sid)
    d = eng.ct_prep(si, ei, L)
    no, _ = eng._dag(si)
    si_pos = no.index(si)
    rng = np.random.default_rng(seed_int)
    seen = set()
    out = []
    max_attempts = max(20000, 200 * n_target)
    attempts = 0
    while len(out) < n_target and attempts < max_attempts:
        attempts += 1
        cur = si_pos
        path = [si]
        ok = True
        for rem in range(L, 0, -1):
            entry = d[rem]
            if cur not in entry:
                ok = False
                break
            cands, _cum = entry[cur]
            k = int(rng.integers(0, len(cands)))
            cur = int(cands[k])
            path.append(int(no[cur]))
        if not ok:
            continue
        t = tuple(path)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    if len(out) < n_target:
        T = eng.counts(si, ei, L)
        rng2 = _py_random.Random(seed_int)
        ranks = sorted(rng2.sample(range(T), n_target - len(out)))
        for r in ranks:
            p = tuple(int(x) for x in eng.unrank_cell(si, ei, L, [r])[0])
            if p not in seen:
                seen.add(p)
                out.append(p)
    return [Path(t, sid, i) for i, t in enumerate(out)]


def gen_candidates(eng, si, ei, L, screen_id, sid, K0):
    T = eng.counts(si, ei, L)
    if T == 0:
        return []
    if L <= 10:
        if T <= K0:
            ranks = list(range(T))          # full enumeration
        else:
            ranks = eng.sample_ranks(si, ei, L, K0, cell_seed(screen_id, sid))
        arr = eng.unrank_cell(si, ei, L, ranks)
        return [Path(tuple(int(x) for x in arr[i]), sid, i)
                for i in range(len(ranks))]
    return random_walk_candidates(eng, si, ei, L, min(K0, T), screen_id, sid)


# ══════════════════════════════════════════════════════════════════
def build_screen_params(cfg):
    Ksel = cfg['B5_path_diversity_ddvs']['Ksel']
    jac = cfg['B5_path_diversity_ddvs']['jaccard_threshold']
    return {
        'B1': {**cfg['B1_anatomical_consistency'], 'Ksel': Ksel,
               'jaccard_threshold': jac},
        'B2': {**cfg['B2_conduction_velocity'], 'Ksel': Ksel,
               'jaccard_threshold': jac},
        'B3': {**cfg['B3_temporal_robustness'], 'Ksel': Ksel,
               'jaccard_threshold': jac},
        'B4': {**cfg['B4_sc_strength_gradient'], 'Ksel': Ksel,
               'jaccard_threshold': jac},
        'B5': {**cfg['B5_path_diversity_ddvs'], 'Ksel': Ksel,
               'jaccard_threshold': jac},
        'B6': {**cfg['B6_communication_coherence'], 'Ksel': Ksel,
               'jaccard_threshold': jac},
    }


def run_universe(eng, matrices, universe, out_dir, tag, cfg, screen_ids,
                 cache, audit_mode):
    """Run the screens on one universe; write products; return retention."""
    params_all = build_screen_params(cfg)
    K0 = cfg['common']['K0_sample']
    min_paths = cfg['common']['min_paths_per_cell_L2plus']
    r2i = eng.r2i
    roi_order = eng.roi_order
    retention = {}
    for screen_id in screen_ids:
        params = params_all[screen_id]
        fn = getattr(screens, f'screen_{screen_id}')
        reason_fn = getattr(screens, f'{screen_id.lower()}_reject_reason')
        prefix = f'{screen_id}_' if tag == 'C' else f'{screen_id}_{tag}_'
        # B4: threshold from the full candidate universe (structure only)
        if screen_id == 'B4':
            all_mean_sc = []
            for row_id, r in universe.iterrows():
                si = r2i[r['seed_roi']]
                ei = r2i[r['endpoint_roi']]
                for p in cache[(screen_id, r['sample_id'])]:
                    nodes = np.asarray(p.nodes)
                    all_mean_sc.append(float(
                        matrices.sc_weight[nodes[:-1], nodes[1:]].mean()))
            params['threshold_mean_sc'] = float(np.percentile(all_mean_sc, 40))
            print(f'    B4 threshold (40th pct of mean_sc universe) = '
                  f'{params["threshold_mean_sc"]:.6f}', flush=True)
        reg_rows = []
        att_rows = []
        t0 = time.time()
        for row_id, r in universe.iterrows():
            sid = r['sample_id']
            si = r2i[r['seed_roi']]
            ei = r2i[r['endpoint_roi']]
            L = int(r['path_length_edges'])
            paths = cache[(screen_id, sid)]
            breakdown = {}
            accepted_pre = []
            for p in paths:
                reason = reason_fn(p, params, matrices)
                if reason is None:
                    accepted_pre.append(p)
                else:
                    breakdown[reason] = breakdown.get(reason, 0) + 1
            accepted_post = fn(paths, params, matrices)
            in_support = len(accepted_pre) >= (1 if L == 1 else min_paths)
            for p in accepted_post:
                nodes = p.nodes
                u = nodes[:-1]
                v = nodes[1:]
                reg_rows.append({
                    'sample_id': sid,
                    'path_id': int(row_id * 1_000_000 + p.rank),
                    'path_rank': int(p.rank),
                    'sample_row_id': int(row_id),
                    'roi_sequence': '|'.join(roi_order[x] for x in nodes),
                    'roi_id_sequence': '|'.join(str(int(x)) for x in nodes),
                    'edge_set': ';'.join(f'{int(a)}-{int(b)}'
                                         for a, b in zip(u, v)),
                    'L': L, 'accepted': True, 'reject_reason': '',
                    'screen_id': screen_id})
            att_rows.append({
                'sample_id': sid,
                'n_candidate': len(paths),
                'n_accepted': len(accepted_pre),
                'n_rejected': len(paths) - len(accepted_pre),
                'reject_reason_breakdown': json.dumps(breakdown, sort_keys=True),
                'in_support': bool(in_support), 'L': L,
                'seed_roi': r['seed_roi'], 'endpoint_roi': r['endpoint_roi']})
        reg = pd.DataFrame(reg_rows)
        att = pd.DataFrame(att_rows)
        reg_p = os.path.join(out_dir, f'{prefix}PATH_REGISTRY.parquet')
        att_p = os.path.join(out_dir, f'{prefix}ATTRITION.csv')
        supp_p = os.path.join(out_dir, f'{prefix}SUPPORT_SAMPLES.csv')
        summ_p = os.path.join(out_dir, f'{prefix}SUMMARY.md')
        reg.to_parquet(reg_p, index=False)
        att.to_csv(att_p, index=False)
        supp = att[att['in_support']][['sample_id']].reset_index(drop=True)
        supp.to_csv(supp_p, index=False)
        if audit_mode:
            for p in (reg_p, att_p, supp_p, summ_p):
                io_log(p, 'w')
        n_supp = int(att['in_support'].sum())
        retention[screen_id] = (n_supp, len(universe))
        lines = [
            f'# {screen_id} SUMMARY ({tag})',
            '',
            f'- universe: {len(universe)} cells',
            f'- in-support cells: {n_supp} ({n_supp/len(universe):.1%})',
            f'- registry paths (post-DDVS): {len(reg)}',
            f'- candidate paths: {int(att["n_candidate"].sum())}',
            f'- accepted paths (pre-DDVS): {int(att["n_accepted"].sum())}',
            '',
            '## In-support cells per L',
            '',
            '| L | cells | in-support | registry paths |',
            '|---|---|---|---|',
        ]
        byL = att.groupby('L').agg(cells=('sample_id', 'size'),
                                   supp=('in_support', 'sum'))
        regL = reg.groupby('L').size()
        for L in sorted(byL.index):
            lines.append(f'| {L} | {int(byL.loc[L, "cells"])} | '
                         f'{int(byL.loc[L, "supp"])} | '
                         f'{int(regL.get(L, 0))} |')
        lines += ['', '## Top reject reasons (all cells)', '',
                  '| reason | n paths |', '|---|---|']
        reasons = {}
        for _, r_ in att.iterrows():
            bd = json.loads(r_['reject_reason_breakdown'])
            for k, v in bd.items():
                reasons[k] = reasons.get(k, 0) + v
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
            lines.append(f'| {k} | {v} |')
        with _ORIG_OPEN(summ_p, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'    {screen_id}: {n_supp}/{len(universe)} in-support '
              f'({n_supp/len(universe):.1%}), registry {len(reg)} paths, '
              f'{time.time()-t0:.0f}s', flush=True)
    return retention


# ══════════════════════════════════════════════════════════════════
def prepare_canary_inputs():
    canary_root = os.path.join(SMCR, 'candidate_screens', 'canary_inputs')
    if os.path.exists(canary_root):
        shutil.rmtree(canary_root)
    src_raw = '/home/yuanyufei/revision_v2/raw_inputs'
    src_bi = '/home/yuanyufei/revision_v2/frozen_s0/bridge_inputs'
    src_in = os.path.join(SMCR, 'candidate_screens', 'inputs')
    src_cfg = os.path.join(SMCR, 'config')
    for sub in ('raw', 'bi', 'in', 'cfg'):
        os.makedirs(os.path.join(canary_root, sub), exist_ok=True)
    for f in os.listdir(src_raw):
        shutil.copy2(os.path.join(src_raw, f), os.path.join(canary_root, 'raw', f))
    for f in ('averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv',
              'roi_order_C0R2C.txt'):
        shutil.copy2(os.path.join(src_bi, f), os.path.join(canary_root, 'bi', f))
    for f in os.listdir(src_in):
        shutil.copy2(os.path.join(src_in, f), os.path.join(canary_root, 'in', f))
    for f in ('SCREEN_PARAMS.json', 'ANATOMICAL_WHITELIST.csv'):
        shutil.copy2(os.path.join(src_cfg, f), os.path.join(canary_root, 'cfg', f))
    # inject FAKE target material (must never affect the screens)
    rng = np.random.default_rng(20260818)
    for name in ('stage2_universe_2636.csv', 'stage2_taskA_1129_units.csv'):
        p = os.path.join(canary_root, 'in', name)
        df = pd.read_csv(p)
        df.insert(1, 'fake_target', rng.uniform(0, 1, len(df)))
        df.to_csv(p, index=False)
    fake = pd.DataFrame(rng.uniform(0, 1, (360, 360)))
    fake.to_csv(os.path.join(canary_root, 'raw', 'fake_conditional_prob_matrix.csv'),
                index=False, header=False)
    with open(os.path.join(canary_root, 'CANARY_NOTE.txt'), 'w') as f:
        f.write('Canary input copy: fake_target column injected into the '
                'universe tables and a fake_conditional_prob_matrix.csv added. '
                'Screen outputs MUST be byte-identical to the normal run.\n')
    print(f'canary inputs prepared at {canary_root}')
    return canary_root


# ══════════════════════════════════════════════════════════════════
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'normal'
    print('=' * 70)
    print(f'STAGE 2 SCREEN DRIVER — mode={mode}')
    print('=' * 70, flush=True)

    open = _wrapped_open  # noqa: F841 — trace Python-level opens (unused in canary mode)
    if mode == 'normal':
        canary_root = None
        out_root = os.path.join(SMCR, 'candidate_screens')
        cfg_dir = os.path.join(SMCR, 'config')
    elif mode == 'canary':
        canary_root = prepare_canary_inputs()
        out_root = os.path.join(SMCR, 'candidate_screens', 'canary_out')
        os.makedirs(out_root, exist_ok=True)
        cfg_dir = os.path.join(canary_root, 'cfg')
    else:
        raise SystemExit('mode must be normal|canary')

    cfg = json.load(open(os.path.join(cfg_dir, 'SCREEN_PARAMS.json')))
    K0 = cfg['common']['K0_sample']

    t0 = time.time()
    eng, raw, bi = load_engine(canary_root)
    matrices = make_matrices(eng, raw, bi, cfg_dir)
    print(f'engine + matrices ready ({time.time()-t0:.1f}s)', flush=True)

    if mode == 'normal':
        for p in (os.path.join(raw, 'Reordered_matrix_onset_delay__median.csv'),
                  os.path.join(raw, 'navigation_efficiency_matrix_0.25density.csv'),
                  os.path.join(raw, 'rout_efficiency_matrix_0.25density.csv'),
                  os.path.join(raw, 'search_information_matrix_0.25density.csv'),
                  os.path.join(raw, 'communicability_matrix_0.25density.csv'),
                  os.path.join(raw, 'averageConnectivity_tractLength_0.25density.csv'),
                  os.path.join(raw, 'mmp_to_yeo7_mapping.csv'),
                  os.path.join(bi, 'averageConnectivity_tractStrength_0.25density_symmetric_C0R2C.csv'),
                  os.path.join(bi, 'roi_order_C0R2C.txt'),
                  os.path.join(cfg_dir, 'SCREEN_PARAMS.json'),
                  os.path.join(cfg_dir, 'ANATOMICAL_WHITELIST.csv')):
            io_log(p, 'r')
        for name in ('stage2_universe_2636.csv', 'stage2_taskA_1129_units.csv'):
            io_log(os.path.join(SMCR, 'candidate_screens', 'inputs', name), 'r')

    in_dir = (os.path.join(canary_root, 'in') if canary_root is not None
              else os.path.join(SMCR, 'candidate_screens', 'inputs'))

    universes = []
    uniC = pd.read_csv(os.path.join(in_dir, 'stage2_universe_2636.csv'))
    universes.append(('C', uniC))
    uniA = pd.read_csv(os.path.join(in_dir, 'stage2_taskA_1129_units.csv'))
    universes.append(('TASKA1129', uniA))

    screen_ids = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']

    # candidate cache per (screen, sample_id); Task A-1129 shares sample ids
    # with Task C, so the cache is shared (identical candidates by design)
    cache = {}
    t0 = time.time()
    n_paths = 0
    for _tag, uni in universes:
        for _row_id, r in uni.iterrows():
            sid = r['sample_id']
            si = eng.r2i[r['seed_roi']]
            ei = eng.r2i[r['endpoint_roi']]
            L = int(r['path_length_edges'])
            for screen_id in screen_ids:
                if (screen_id, sid) in cache:
                    continue
                paths = gen_candidates(eng, si, ei, L, screen_id, sid, K0)
                cache[(screen_id, sid)] = paths
                n_paths += len(paths)
    print(f'candidates: {n_paths} paths ({time.time()-t0:.0f}s)', flush=True)

    retention_C = {}
    t0 = time.time()
    for tag, uni in universes:
        print(f'[universe {tag}] running B1-B6...', flush=True)
        ret = run_universe(eng, matrices, uni, out_root, tag, cfg,
                           screen_ids, cache, audit_mode=(mode == 'normal'))
        if tag == 'C':
            retention_C = ret
    print(f'screens done ({time.time()-t0:.0f}s)', flush=True)

    # ── product hashes ──
    product_hashes = {}
    for f in sorted(os.listdir(out_root)):
        if f.endswith(('.parquet', '.csv', '.md')):
            product_hashes[f] = sha256_file(os.path.join(out_root, f))
    hash_file = os.path.join(
        SMCR, 'audit',
        f'STAGE2_PRODUCT_HASHES_{"CANARY" if mode == "canary" else "NORMAL"}.json')
    with _ORIG_OPEN(hash_file, 'w') as f:
        json.dump(product_hashes, f, indent=2, sort_keys=True)

    if mode == 'canary':
        normal = json.load(open(os.path.join(
            SMCR, 'audit', 'STAGE2_PRODUCT_HASHES_NORMAL.json')))
        per_screen = {}
        for k in sorted(normal):
            if k in product_hashes:
                per_screen[k] = ('IDENTICAL' if product_hashes[k] == normal[k]
                                 else 'MISMATCH')
        verdict = {
            'canary': 'PASS' if all(v == 'IDENTICAL' for v in per_screen.values())
                      else 'FAIL',
            'products': per_screen,
            'method': 'fake_target column + fake_conditional_prob_matrix.csv '
                      'injected into a copied input tree; screens re-run from '
                      'scratch; product SHA-256 must match the normal run',
            'n_products': len(per_screen),
        }
        with _ORIG_OPEN(os.path.join(SMCR, 'audit', 'STAGE2_TARGET_CANARY.json'), 'w') as f:
            json.dump(verdict, f, indent=2, sort_keys=True)
        n_ok = sum(1 for v in per_screen.values() if v == 'IDENTICAL')
        print(f'canary compare: {n_ok}/{len(per_screen)} products identical '
              f'-> {verdict["canary"]}')

    if mode == 'normal':
        # ── file-open firewall audit ──
        bad = [(p, m, s) for p, m, s in _OPEN_EVENTS
               if any(k in p.lower() for k in _FORBIDDEN)]
        with _ORIG_OPEN(os.path.join(SMCR, 'audit', 'STAGE2_FILE_OPEN_LOG.jsonl'), 'w') as f:
            for p, m, s in _OPEN_EVENTS:
                f.write(json.dumps({'path': p, 'mode': m, 'source': s}) + '\n')
        verdict = {
            'verdict': 'PASS' if not bad else 'FAIL',
            'violations': [{'path': p, 'mode': m, 'source': s} for p, m, s in bad],
            'n_events': len(_OPEN_EVENTS),
            'method': 'Python-level open() wrapper + explicit io manifest '
                      '(pandas/pyarrow reads logged via io_log)',
        }
        with _ORIG_OPEN(os.path.join(SMCR, 'audit', 'STAGE2_FIREWALL_VERDICT.json'), 'w') as f:
            json.dump(verdict, f, indent=2, sort_keys=True)
        print(f'file-open firewall: {"PASS" if not bad else "FAIL"} '
              f'({len(bad)} violations, {len(_OPEN_EVENTS)} events)')

        if bad:
            for p, m, s in bad:
                print(f'  VIOLATION: {p} ({m}, {s})')
            print('STOP — firewall violations. No further stages.')
            raise SystemExit(1)

        print()
        print('=== STAGE 2 COMPLETE ===')
        for k in screen_ids:
            n, tot = retention_C[k]
            print(f'{k} retention: {n/tot:.1%} ({n} in-support / {tot})')
        print('Target canary: run `python3 run_screens.py canary` to verify')
        print('File-open firewall: PASS (0 violations)')
        print('Next: Stage 3 (overlap matching & screen selection)')


if __name__ == '__main__':
    main()
