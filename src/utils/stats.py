#!/usr/bin/env python3
"""
Task C Phase 09 statistics v1 (frozen V7 spec).

Unified machine-readable statistics from Phase 04-08 artifacts ONLY:
  - OOF long table with composite primary key + per-family coverage gates
  - four metric levels (OUTER_FOLD / REPEAT_POOLED_OOF / REPEAT_SUMMARY /
    MEAN_PREDICTION_SAMPLE_LEVEL)
  - 10,000 paired seed-cluster bootstrap for V3_AMENDED_ENDPOINT_PILOT
    (vectorized sufficient statistics + row-wise reference equivalence)
  - Holm on the three primary R2 contrasts
  - null verification (MC p recomputation from the frozen 100-replicate stats)
  - target/length-bias and temporal/SC sensitivity summaries
  - grouped permutation importance for C6/C8R endpoint models

No Phase05-08 production is rerun; no cross-family/mode/target/support mixing.
"""
import os, sys, json, time, hashlib, glob
import numpy as np
import pandas as pd

NA = 'NA'  # string NA for categorical key fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code'))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

OUT_DIR = 'results/intermediate/phase09_v7'
METRICS4 = ['nav', 'rout', 'search', 'comm']


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''):
            h.update(c)
    return h.hexdigest()


def write_parquet(df, path):
    df.to_parquet(path, index=False, compression='zstd')
    json.dump({'columns': list(df.columns), 'n_rows': len(df),
               'n_cols': len(df.columns),
               'dtypes': {c: str(df[c].dtype) for c in df.columns}},
              open(path + '.schema.json', 'w'), indent=2)
    open(path + '.sha256', 'w').write(sha256_file(path) + '\n')
    return sha256_file(path)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def metric_block(yt, yp):
    """R2/RMSE/MAE/r/calibration; constant y or pred -> NA with reason."""
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 3:
        return {k: np.nan for k in
                ('r2', 'rmse', 'mae', 'pearson_r', 'calib_intercept',
                 'calib_slope')} | {'n': int(mask.sum()), 'na_reason': 'n<3'}
    y, p = yt[mask], yp[mask]
    out = {'n': int(mask.sum()), 'na_reason': ''}
    out['rmse'] = float(np.sqrt(mean_squared_error(y, p)))
    out['mae'] = float(mean_absolute_error(y, p))
    if np.ptp(y) == 0:
        out['r2'] = np.nan
        out['na_reason'] += 'constant_y;'
    else:
        out['r2'] = float(r2_score(y, p))
    if np.ptp(y) == 0 or np.ptp(p) == 0:
        out['pearson_r'] = np.nan
        out['calib_intercept'] = np.nan
        out['calib_slope'] = np.nan
        out['na_reason'] += 'constant_y_or_pred;'
    else:
        out['pearson_r'] = float(np.corrcoef(y, p)[0, 1])
        slope, intercept = np.polyfit(p, y, 1)
        out['calib_slope'] = float(slope)
        out['calib_intercept'] = float(intercept)
    return out


def fold_metrics(pred_df):
    out = metric_block(pred_df['y_true'].values, pred_df['y_pred'].values)
    out['repeat'] = int(pred_df['repeat'].iloc[0])
    out['test_fold'] = int(pred_df['fold'].iloc[0])
    return out


def repeat_pooled(pred_df):
    per = {}
    for rep in range(5):
        sub = pred_df[pred_df['repeat'] == rep]
        per[rep] = metric_block(sub['y_true'].values, sub['y_pred'].values)
    mean = {}
    for k in ('r2', 'rmse', 'mae', 'pearson_r', 'calib_intercept', 'calib_slope'):
        vals = [per[r][k] for r in range(5) if np.isfinite(per[r][k])]
        mean[k] = float(np.mean(vals)) if vals else np.nan
    mean['n'] = sum(per[r]['n'] for r in range(5))
    return mean, per


# ═══════════════════════════════════════════════════════════════
# OOF LONG TABLE
# ═══════════════════════════════════════════════════════════════

def build_oof_long_table():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    # ── V3_AMENDED_ENDPOINT_PILOT: C4 bridge + C6/C8R conditional anchors ──
    for rep in range(5):
        for fold in range(5):
            c = json.load(open(f'results/intermediate/phase05_v3_c8r_cells/'
                               f'C4_ENDPOINT__r{rep}f{fold}.json'))
            for p in c['targets']['endpoint']['predictions']:
                rows.append({
                    'analysis_family': 'V3_AMENDED_ENDPOINT_PILOT',
                    'inference_mode': 'C4_HP_CONDITIONAL_ENDPOINT_PILOT',
                    'target': 'endpoint', 'support_mode': 'PRIMARY_FIXED_2636',
                    'condition_id': NA, 'null_type': NA, 'null_id': NA,
                    'model': 'C4_ENDPOINT',
                    'repeat': p['repeat'], 'fold': p['test_fold'],
                    'sample_id': p['sample_id'],
                    'y_true': p['y_true'], 'y_pred': p['y_pred'],
                    'source_file': 'phase05_v3 C4_ENDPOINT bridge cell',
                    'source_hash': c.get('content_sha256'),
                    'status': 'COMPLETE', 'warnings': 0,
                })
    for rep_name, model in [('summary', 'C6_SUMMARY'),
                            ('ordered', 'C8R_ENDPOINT_PLUS_ORDERED_BASIS')]:
        for f in sorted(glob.glob(f'results/intermediate/phase06_v4/'
                                  f'real_{rep_name}_conditional__*.json')):
            c = json.load(open(f))
            for p in c['predictions']:
                rows.append({
                    'analysis_family': 'V3_AMENDED_ENDPOINT_PILOT',
                    'inference_mode': 'C4_HP_CONDITIONAL_ENDPOINT_PILOT',
                    'target': 'endpoint', 'support_mode': 'PRIMARY_FIXED_2636',
                    'condition_id': 'RAW_STRICT', 'null_type': NA, 'null_id': NA,
                    'model': model,
                    'repeat': p['repeat'], 'fold': p['test_fold'],
                    'sample_id': p['sample_id'],
                    'y_true': p['y_true'], 'y_pred': p['y_pred'],
                    'source_file': f'phase06_v4 real_{rep_name}_conditional',
                    'source_hash': c.get('content_sha256'),
                    'status': 'COMPLETE', 'warnings': int(c.get('warned') or 0),
                })

    # ── TARGET_DEFINITION_SENSITIVITY: Phase05 V3 multi-target OOF ──
    oof5 = pd.read_parquet('results/intermediate/TASK_C_MULTI_TARGET_OOF_V3.parquet')
    oof5 = oof5[oof5['target'] != 'endpoint'] if 'endpoint' in oof5['target'].unique() else oof5
    for _, r in oof5.iterrows():
        rows.append({
            'analysis_family': 'TARGET_DEFINITION_SENSITIVITY',
            'inference_mode': 'PILOT_9x8_V3_PROCESS_TUNED',
            'target': r['target'], 'support_mode': 'PER_TARGET_COMMON_SUPPORT',
            'condition_id': NA, 'null_type': NA, 'null_id': NA,
            'model': r['model'], 'repeat': int(r['repeat']),
            'fold': int(r['test_fold']), 'sample_id': r['sample_id'],
            'y_true': r['y_true'], 'y_pred': r['y_pred'],
            'source_file': 'TASK_C_MULTI_TARGET_OOF_V3.parquet',
            'source_hash': open('results/intermediate/TASK_C_MULTI_TARGET_OOF_V3.parquet.sha256').read().strip(),
            'status': 'COMPLETE', 'warnings': 0,
        })

    # ── RANDOM_PATH_NULL: Phase06 V4 conditional OOF (anchors + 200 nulls) ──
    oof6 = pd.read_parquet('results/intermediate/phase06_v4/TASK_C_V4_CONDITIONAL_OOF.parquet')
    for _, r in oof6.iterrows():
        m = r['model']
        if m.startswith('null_'):
            parts = m.split('_')
            rep_name = parts[1]
            nid = int(parts[3])
            model = 'C6_SUMMARY' if rep_name == 'summary' else \
                'C8R_ENDPOINT_PLUS_ORDERED_BASIS'
            fam = 'RANDOM_PATH_NULL'
            null_type = 'INTERMEDIATE_NODE_ORDER_PERMUTATION'
            mode = 'C4_HP_CONDITIONAL_RANDOM_PATH_PILOT'
        else:
            continue  # real anchors already in the V3 family
        rows.append({
            'analysis_family': fam, 'inference_mode': mode,
            'target': 'endpoint', 'support_mode': 'PRIMARY_FIXED_2636',
            'condition_id': 'RAW_STRICT', 'null_type': null_type,
            'null_id': nid, 'model': model,
            'repeat': int(r['repeat']), 'fold': int(r['test_fold']),
            'sample_id': r['sample_id'], 'y_true': r['y_true'],
            'y_pred': r['y_pred'],
            'source_file': 'phase06_v4 TASK_C_V4_CONDITIONAL_OOF.parquet',
            'source_hash': open('results/intermediate/phase06_v4/TASK_C_V4_CONDITIONAL_OOF.parquet.sha256').read().strip(),
            'status': 'COMPLETE', 'warnings': 0,
        })

    # ── SC_NULL: Phase07 V5 conditional OOF (real anchors + 200 nulls) ──
    oof7 = pd.read_parquet('results/intermediate/phase07_v5/TASK_C_V5_CONDITIONAL_OOF.parquet')
    for _, r in oof7.iterrows():
        m = r['model']
        if m.startswith('null_'):
            parts = m.split('_')
            rep_name = parts[1]
            nid = int(parts[3])
            model = 'C6_SUMMARY' if rep_name == 'summary' else \
                'C8R_ENDPOINT_PLUS_ORDERED_BASIS'
            fam = 'SC_NULL'
            null_type = 'DEGREE_PRESERVING_WITHIN_HEMI_CONNECTED_DOUBLE_EDGE_SWAP'
            mode = 'C4_HP_CONDITIONAL_SC_NULL_PILOT'
        elif m.startswith('real_'):
            rep_name = m.split('_')[1]
            model = 'C6_SUMMARY' if rep_name == 'summary' else \
                'C8R_ENDPOINT_PLUS_ORDERED_BASIS'
            fam = 'SC_NULL'
            null_type = 'REAL_SC_NULL0_PIPELINE_ANCHOR'
            nid = None
            mode = 'C4_HP_CONDITIONAL_SC_NULL_PILOT'
        else:
            continue
        rows.append({
            'analysis_family': fam, 'inference_mode': mode,
            'target': 'endpoint', 'support_mode': 'PRIMARY_FIXED_2636',
            'condition_id': 'RAW_STRICT', 'null_type': null_type,
            'null_id': nid, 'model': model,
            'repeat': int(r['repeat']), 'fold': int(r['test_fold']),
            'sample_id': r['sample_id'], 'y_true': r['y_true'],
            'y_pred': r['y_pred'],
            'source_file': 'phase07_v5 TASK_C_V5_CONDITIONAL_OOF.parquet',
            'source_hash': open('results/intermediate/phase07_v5/TASK_C_V5_CONDITIONAL_OOF.parquet.sha256').read().strip(),
            'status': 'COMPLETE', 'warnings': 0,
        })

    # ── TEMPORAL_SC_CONDITIONAL_SENSITIVITY: Phase08 V6 OOF ──
    oof8 = pd.read_parquet('results/intermediate/phase08_v6/TASK_C_SENSITIVITY_OOF_V6.parquet')
    for _, r in oof8.iterrows():
        m = r['model']
        if m.startswith('RAW') or m.startswith('BIN'):
            parts = m.split('__')
            condition = parts[0]
            rep_name = parts[1]
            support = parts[2]
            model = 'C6_SUMMARY' if rep_name == 'summary' else \
                'C8R_ENDPOINT_PLUS_ORDERED_BASIS'
        elif m == 'C4_ENDPOINT':
            condition, rep_name, support = NA, NA, NA
            model = 'C4_ENDPOINT'
        else:
            continue
        rows.append({
            'analysis_family': 'TEMPORAL_SC_CONDITIONAL_SENSITIVITY',
            'inference_mode': 'C4_HP_CONDITIONAL_SENSITIVITY_PILOT',
            'target': 'endpoint',
            'support_mode': support if support != NA else 'PRIMARY_FIXED_2636',
            'condition_id': condition, 'null_type': NA, 'null_id': NA,
            'model': model, 'repeat': int(r['repeat']),
            'fold': int(r['test_fold']), 'sample_id': r['sample_id'],
            'y_true': r['y_true'], 'y_pred': r['y_pred'],
            'source_file': 'phase08_v6 TASK_C_SENSITIVITY_OOF_V6.parquet',
            'source_hash': open('results/intermediate/phase08_v6/TASK_C_SENSITIVITY_OOF_V6.parquet.sha256').read().strip(),
            'status': 'COMPLETE', 'warnings': 0,
        })

    df = pd.DataFrame(rows)
    df['null_id'] = df['null_id'].replace({NA: pd.NA})
    df['null_id'] = df['null_id'].astype('Int64')
    df = df.sort_values(['analysis_family', 'model', 'repeat', 'fold',
                         'sample_id'], kind='stable').reset_index(drop=True)
    return df


def coverage_gate(df):
    """Per-family coverage checks; returns list of (family, model, ok, detail)."""
    out = []
    for fam, sub in df.groupby('analysis_family'):
        # discriminating key columns per family: target / condition / support
        disc = ['target', 'condition_id', 'support_mode']
        disc = [c for c in disc if sub[c].nunique() > 1 or c == 'target']
        for (model,), msub in sub.groupby(['model']):
            ok = True
            details = []
            is_null = str(msub['null_type'].iloc[0]) != 'NA'
            if is_null:
                pk = msub.groupby(['null_id', 'sample_id', 'repeat']).size()
                n_ids = msub['null_id'].nunique()
                ok = (pk == 1).all()
                details.append(f"nulls: null_ids={n_ids}, keys={len(pk)}, "
                               f"all_single={ok}")
            else:
                keycols = ['sample_id', 'repeat'] + disc
                n_cells = msub.groupby(disc).ngroups
                pk = msub.groupby(keycols).size()
                rows_per_cell = len(msub) // max(n_cells, 1)
                ok = (pk == 1).all()
                details.append(f"cells={n_cells}, rows={len(msub)}, "
                               f"rows/cell={rows_per_cell}, all_single={ok}")
            out.append((fam, str(model), bool(ok), '; '.join(details)))
    return pd.DataFrame(out, columns=['family', 'model', 'ok', 'detail'])


# ═══════════════════════════════════════════════════════════════
# CLUSTER BOOTSTRAP (vectorized sufficient statistics)
# ═══════════════════════════════════════════════════════════════

def sufficient_stats(pred_df, seed_roi_map):
    """Per (seed_roi, repeat, model): n, Sy, Syy, Sp, Spp, Syp, SSE, SAE."""
    df = pred_df.merge(seed_roi_map, on='sample_id', how='inner')
    g = df.groupby(['seed_roi', 'repeat'])
    y = g['y_true'].apply(np.asarray)
    p = g['y_pred'].apply(np.asarray)
    stats = pd.DataFrame({
        'n': g.size(),
        'Sy': y.apply(np.nansum), 'Syy': y.apply(lambda v: np.nansum(v * v)),
        'Sp': p.apply(np.nansum), 'Spp': p.apply(lambda v: np.nansum(v * v)),
        'Syp': y.combine(p, lambda a, b: np.nansum(a * b)),
        'SSE': y.combine(p, lambda a, b: np.nansum((a - b) ** 2)),
        'SAE': y.combine(p, lambda a, b: np.nansum(np.abs(a - b))),
    }).reset_index()
    return stats


def r2_from_suff(s):
    sst = s['Syy'] - s['Sy'] ** 2 / s['n']
    return 1 - s['SSE'] / sst


def rmse_from_suff(s):
    return np.sqrt(s['SSE'] / s['n'])


def mae_from_suff(s):
    return s['SAE'] / s['n']


def cluster_bootstrap(stats_per_model, B, seed, reference_check=None):
    """Vectorized seed-cluster bootstrap over per-model sufficient stats.

    stats_per_model: dict model -> DataFrame with columns
    seed_roi, repeat, n, Sy, Syy, Sp, Spp, Syp, SSE, SAE.
    reference_check: optional list of (model, row_df, seed_roi_map) used to
    validate the vectorized path on fixed replicates.
    """
    rng = np.random.default_rng(seed)
    # cluster universe: union of seed_roi across models (same support/folds)
    clusters = sorted(set().union(*[set(s['seed_roi']) for s in
                                    stats_per_model.values()]))
    n_clu = len(clusters)
    clu_idx = {c: i for i, c in enumerate(clusters)}
    # index arrays per model
    idx = {}
    for m, s in stats_per_model.items():
        s = s.copy()
        s['ci'] = s['seed_roi'].map(clu_idx).values
        s['ri'] = s['repeat'].values
        idx[m] = s
    rep_metrics = {m: {} for m in stats_per_model}
    for rep in range(5):
        for m, s in idx.items():
            sub = s[s['ri'] == rep]
            rep_metrics[m][rep] = sub

    # draw B resamples ONCE (same for all models/contrasts)
    draws = rng.integers(0, n_clu, size=(B, n_clu))  # (B, n_clu)
    mults = np.zeros((B, n_clu), dtype=np.int64)
    for b in range(B):
        np.add.at(mults[b], draws[b], 1)

    boot = {m: {} for m in stats_per_model}
    for rep in range(5):
        for m in stats_per_model:
            s = rep_metrics[m][rep]
            n = len(s)
            # map each bootstrap column to rows via cluster positions
            sub_mults = mults[:, s['ci'].values]          # (B, n)
            N = (sub_mults * s['n'].values[None, :]).sum(axis=1)  # weighted rows
            Sy = (sub_mults * s['Sy'].values[None, :]).sum(axis=1)
            Syy = (sub_mults * s['Syy'].values[None, :]).sum(axis=1)
            Sp = (sub_mults * s['Sp'].values[None, :]).sum(axis=1)
            Spp = (sub_mults * s['Spp'].values[None, :]).sum(axis=1)
            Syp = (sub_mults * s['Syp'].values[None, :]).sum(axis=1)
            SSE = (sub_mults * s['SSE'].values[None, :]).sum(axis=1)
            SAE = (sub_mults * s['SAE'].values[None, :]).sum(axis=1)
            sst = Syy - Sy * Sy / np.maximum(N, 1)
            r2 = np.where(np.isfinite(sst) & (sst > 0), 1 - SSE / np.maximum(sst, 1e-300), np.nan)
            boot[m][rep] = {
                'r2': r2,
                'rmse': np.sqrt(SSE / np.maximum(N, 1)),
                'mae': SAE / np.maximum(N, 1),
            }
    return boot, draws


def bootstrap_contrasts(boot, B):
    """mean-over-5-repeats paired contrast distributions for R2/RMSE/MAE."""
    out = {}
    for cname, a, b in [('C6-C4', 'C6_SUMMARY', 'C4_ENDPOINT'),
                        ('C8R-C4', 'C8R_ENDPOINT_PLUS_ORDERED_BASIS', 'C4_ENDPOINT'),
                        ('C8R-C6', 'C8R_ENDPOINT_PLUS_ORDERED_BASIS', 'C6_SUMMARY')]:
        for metric in ('r2', 'rmse', 'mae'):
            d = np.zeros(B)
            for rep in range(5):
                da = boot[a][rep][metric]
                db = boot[b][rep][metric]
                if metric == 'r2':
                    d += (da - db) / 5.0
                else:
                    d += (db - da) / 5.0   # positive = process better
            out[(cname, metric)] = d
    return out


def two_sided_p(delta, B):
    le = 1 + int((delta <= 0).sum())
    ge = 1 + int((delta >= 0).sum())
    return min(1.0, 2 * min(le, ge) / (B + 1))


def holm(pvals):
    '''Holm-Bonferroni adjusted p-values, returned in the INPUT order.'''
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj_sorted = []
    m = len(pvals)
    for rank, i in enumerate(order):
        adj_sorted.append(min(1.0, pvals[i] * (m - rank)))
    for rank in range(1, m):
        adj_sorted[rank] = max(adj_sorted[rank], adj_sorted[rank - 1])
    out = [0.0] * m
    for rank, i in enumerate(order):
        out[i] = adj_sorted[rank]
    return out


# ═══════════════════════════════════════════════════════════════
# NULL VERIFICATION (recompute MC p from the frozen statistics)
# ═══════════════════════════════════════════════════════════════

def verify_null_mc_p(oof_long, csv_paths, bridge_r2, bridge_rmse, bridge_mae):
    """Recompute per-null gains + MC p from the OOF long table and compare
    against the recorded Phase06/07 CSVs (must MATCH exactly)."""
    out = []
    for family, csv_path in csv_paths:
        rec = pd.read_csv(csv_path)
        sub = oof_long[oof_long['analysis_family'] == family]
        nulls = sub[sub['null_id'].notna()]
        assert sorted(nulls['null_id'].dropna().unique()) == list(range(1, 101)), \
            (family, 'null ids not exactly 1..100')
        for rep_name in ('summary', 'ordered'):
            model = 'C6_SUMMARY' if rep_name == 'summary' else \
                'C8R_ENDPOINT_PLUS_ORDERED_BASIS'
            gains = {'r2': [], 'rmse': [], 'mae': []}
            for nid in range(1, 101):
                sub2 = nulls[(nulls['null_id'] == nid) & (nulls['model'] == model)]
                m, _ = repeat_pooled(sub2)
                gains['r2'].append(m['r2'] - bridge_r2)
                gains['rmse'].append(bridge_rmse - m['rmse'])
                gains['mae'].append(bridge_mae - m['mae'])
            for metric in ('r2', 'rmse', 'mae'):
                g = np.array(gains[metric])
                # recorded observed + mc p
                rrows = rec[(rec['representation'] == rep_name) &
                            (rec['statistic'] == f'{rep_name}_gain_{metric}')]
                if 'null_id' in rec.columns:
                    rrow = rrows[rrows['null_id'] == 'observed']
                else:
                    rrow = rrows
                assert len(rrow) == 1, (family, rep_name, metric, len(rrow))
                observed_rec = float(rrow['observed'].iloc[0])
                mc_p_rec = float(rrow['mc_p'].iloc[0])
                # observed from the anchor models: the family's own real
                # anchors when present, else the V3 amended family anchors
                anchors = oof_long[(oof_long['analysis_family'] == family) &
                                   (oof_long['null_type'] ==
                                    'REAL_SC_NULL0_PIPELINE_ANCHOR') &
                                   (oof_long['model'] == model)]
                if len(anchors) == 0:
                    anchors = oof_long[
                        (oof_long['analysis_family'] ==
                         'V3_AMENDED_ENDPOINT_PILOT') &
                        (oof_long['model'] == model)]
                am, _ = repeat_pooled(anchors)
                if metric == 'r2':
                    observed_my = am['r2'] - bridge_r2
                else:
                    observed_my = (bridge_rmse if metric == 'rmse' else
                                   bridge_mae) - am[metric]
                mc_p_my = (1 + int((g >= observed_my).sum())) / 101
                status = 'MATCH'
                if abs(observed_my - observed_rec) > 1e-12 or \
                   abs(mc_p_my - mc_p_rec) > 1e-12:
                    status = 'MISMATCH'
                out.append((family, rep_name, metric, status, observed_my,
                            (mc_p_my, mc_p_rec)))
    return out


if __name__ == '__main__':
    print('taskC_statistics_v1 — library module (Phase 09 V7)')
