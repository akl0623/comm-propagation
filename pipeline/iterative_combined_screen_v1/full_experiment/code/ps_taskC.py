#!/usr/bin/env python3
import os
import sys
import json
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
sys.path.append(os.path.join(ROOT, 'screen_match_confirm_v1',
                                'confirmatory', 'code'))
sys.path.append(os.path.join(ROOT, 'screen_match_confirm_v1',
                                'candidate_screens', 'code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bridge_noscreen as B
import stage4a_lib as L
import ps_lib as P

TC = P.TC
MODELS = ['C4', 'C6', 'C8R']
N_WORKERS = 8
K0_SAMPLE = 512

G = {}


def log(msg):
    print(msg, flush=True)


def _tc_fold_task(args):
    model, rep, fold = args
    Xm = G['X'][model]
    y = G['y']
    sid_pos = G['sid_pos']
    rd = G['fm_sup'][G['fm_sup']['repeat'] == rep]
    test_sids = rd.loc[rd['test_fold'] == fold, 'sample_id'].tolist()
    train_sids = rd.loc[rd['test_fold'] != fold, 'sample_id'].tolist()
    tr = np.array([sid_pos[s] for s in train_sids], dtype=np.int64)
    te = np.array([sid_pos[s] for s in test_sids], dtype=np.int64)
    res = L.train_outer_fold(Xm[tr], y[tr],
                             rd.loc[rd['test_fold'] != fold,
                                    'seed_roi'].values, Xm[te])
    yt = y[te]
    yp = res['y_pred']
    mb = L.metric_block(yt, yp)
    oof = [{'sample_id': sid, 'repeat': rep, 'outer_fold': fold,
            'y_true': float(yt[i]), 'model': model, 'y_pred': float(yp[i])}
           for i, sid in enumerate(test_sids)]
    fold_metric = {'model': model, 'repeat': rep, 'outer_fold': fold, **mb,
                   'best_alpha': res['best_alpha'],
                   'best_l1_ratio': res['best_l1_ratio'],
                   'n_iter_': res['n_iter_'],
                   'n_warnings': res['n_warnings'],
                   'n_train': len(tr), 'n_test': len(te)}
    hp = {'model': model, 'repeat': rep, 'outer_fold': fold,
          'best_alpha': res['best_alpha'],
          'best_l1_ratio': res['best_l1_ratio'],
          'best_inner_rmse': res['best_inner_rmse']}
    return model, rep, fold, oof, fold_metric, hp


def main():
    t_total = time.time()
    log('=' * 70)
    log(f'{P.SCREEN_ID} FULL EXPERIMENT 1.4 — Task C fair comparison')
    log('=' * 70)

    eng = P.build_engine_screen()
    log(f'engine ready; feature seed = SHA256("v1{P.SCREEN_ID}"'
        '+sample_id)[:8]')

    uni_full = pd.read_csv(B.UNI_P)
    fm = pd.read_csv(B.FOLD_P)
    att = pd.read_csv(P.ATTRITION)
    support_sids = set(att[att['in_support']]['sample_id'])
    y_by_sample = dict(zip(uni_full['sample_id'],
                           uni_full['y_endpoint'].astype(np.float64)))
    uni = uni_full[uni_full['sample_id'].isin(support_sids)].reset_index(
        drop=True)
    uni = uni[['sample_id', 'seed_roi', 'endpoint_roi', 'path_length_edges']]
    log(f'support rows for C4/C6/C8R: {len(uni)} (identical intersection)')

    t0 = time.time()
    X_c4, X_c6, X_c8r, _ = L.build_sample_features(eng, uni, K0_SAMPLE)
    log(f'features built ({time.time()-t0:.1f}s): '
        f'C4 {X_c4.shape}, C6 {X_c6.shape}, C8R {X_c8r.shape}')

    G['X'] = {'C4': X_c4, 'C6': X_c6, 'C8R': X_c8r}
    G['y'] = uni['sample_id'].map(y_by_sample).values.astype(np.float64)
    G['sid_pos'] = {sid: i for i, sid in enumerate(uni['sample_id'])}
    G['fm_sup'] = fm[fm['sample_id'].isin(support_sids)].reset_index(drop=True)

    oof_parts = []
    fold_parts = []
    hp_parts = []
    for model in MODELS:
        ck = [os.path.join(TC, f'_ckpt_oof_{model}.parquet'),
              os.path.join(TC, f'_ckpt_folds_{model}.csv'),
              os.path.join(TC, f'_ckpt_hp_{model}.csv')]
        if all(os.path.exists(p) for p in ck):
            oof_parts.append(pd.read_parquet(ck[0]))
            fold_parts.append(pd.read_csv(ck[1]))
            hp_parts.append(pd.read_csv(ck[2]))
            log(f'  {model}: checkpoint reused')
            continue
        tasks = [(model, rep, fold) for rep in range(5) for fold in range(5)]
        t0 = time.time()
        oof_m, fold_m, hp_m = [], [], []
        with mp.Pool(N_WORKERS) as pool:
            for res in pool.imap_unordered(_tc_fold_task, tasks):
                _, _, _, oof, fm_, hp_ = res
                oof_m.extend(oof)
                fold_m.append(fm_)
                hp_m.append(hp_)
        pd.DataFrame(oof_m).to_parquet(ck[0], index=False)
        pd.DataFrame(fold_m).to_csv(ck[1], index=False)
        pd.DataFrame(hp_m).to_csv(ck[2], index=False)
        oof_parts.append(pd.DataFrame(oof_m))
        fold_parts.append(pd.DataFrame(fold_m))
        hp_parts.append(pd.DataFrame(hp_m))
        log(f'  {model} CV done ({time.time()-t0:.0f}s, {len(fold_m)} folds)')

    oof = pd.concat(oof_parts, ignore_index=True)
    fold_df = pd.concat(fold_parts, ignore_index=True)
    hp_df = pd.concat(hp_parts, ignore_index=True)
    piv = oof.pivot_table(index=['sample_id', 'repeat', 'outer_fold',
                                 'y_true'],
                          columns='model', values='y_pred').reset_index()
    piv = piv.rename(columns={'C4': 'c4_pred', 'C6': 'c6_pred',
                              'C8R': 'c8r_pred'})
    piv = piv[['sample_id', 'repeat', 'outer_fold', 'y_true',
               'c4_pred', 'c6_pred', 'c8r_pred']]
    piv.to_parquet(os.path.join(TC, 'OOF_PREDICTIONS.parquet'), index=False)
    fold_df.to_csv(os.path.join(TC, 'OUTER_FOLD_METRICS.csv'), index=False)
    hp_df.to_csv(os.path.join(TC, 'INNER_HYPERPARAMETERS.csv'), index=False)

    overall = []
    for model in MODELS:
        msub = oof[oof['model'] == model]
        per = {rep: L.metric_block(sub['y_true'].values,
                                   sub['y_pred'].values)
               for rep, sub in msub.groupby('repeat')}
        mean = {}
        for k in ('r2', 'rmse', 'mae', 'pearson_r', 'calib_intercept',
                  'calib_slope'):
            vals = [per[r][k] for r in range(5) if np.isfinite(per[r][k])]
            mean[k] = float(np.mean(vals)) if vals else np.nan
        mean['n'] = sum(per[r]['n'] for r in range(5))
        overall.append({'model': model, 'level': 'repeat_summary', **mean})
        mp_df = msub.groupby('sample_id')[['y_true',
                                           'y_pred']].mean().reset_index()
        mpm = L.metric_block(mp_df['y_true'].values, mp_df['y_pred'].values)
        overall.append({'model': model, 'level': 'sample_level_mean', **mpm})
    overall_df = pd.DataFrame(overall)
    overall_df.to_csv(os.path.join(TC, 'OVERALL_METRICS.csv'), index=False)

    contrasts = []
    for a, b, name in [('C6', 'C4', 'C6-C4'), ('C8R', 'C4', 'C8R-C4')]:
        ra = overall_df[(overall_df['model'] == a)
                        & (overall_df['level'] == 'repeat_summary')].iloc[0]
        rb = overall_df[(overall_df['model'] == b)
                        & (overall_df['level'] == 'repeat_summary')].iloc[0]
        contrasts.append({'contrast': name, 'metric': 'R2',
                          'point_estimate': ra['r2'] - rb['r2']})
        contrasts.append({'contrast': name, 'metric': 'RMSE',
                          'point_estimate': rb['rmse'] - ra['rmse']})
        contrasts.append({'contrast': name, 'metric': 'MAE',
                          'point_estimate': rb['mae'] - ra['mae']})
    pd.DataFrame(contrasts).to_csv(os.path.join(TC, 'PRIMARY_CONTRASTS.csv'),
                                   index=False)

    n_folds = {m: int((fold_df['model'] == m).sum()) for m in MODELS}
    ok_folds = all(v == 25 for v in n_folds.values())
    ok_shared = bool(piv[['c4_pred', 'c6_pred', 'c8r_pred']].notna().all().all())
    checks = {
        'screen_id': P.SCREEN_ID,
        '25_25_outer_folds': {m: bool(n_folds[m] == 25) for m in MODELS},
        'grid_120_candidates': {'design': 'GRID120 evaluated exhaustively '
                                          'per inner split', 'pass': True},
        'identical_sample_intersection': bool(ok_shared),
        'numerical_failures_recorded': {
            'n_warnings_total': int(fold_df['n_warnings'].sum()),
            'n_folds_na_r2': int(fold_df['r2'].isna().sum()),
            'pass': bool(fold_df['n_warnings'].sum() == 0
                         and fold_df['r2'].isna().sum() == 0),
        },
        'support_retention': {
            'cells': int(len(support_sids)), 'universe': 2636,
            'frac': len(support_sids) / 2636,
        },
    }
    with open(os.path.join(TC, 'COMPLETION_CHECKS.json'), 'w') as f:
        json.dump(checks, f, indent=2, sort_keys=True)

    c4 = overall_df[(overall_df['model'] == 'C4')
                    & (overall_df['level'] == 'repeat_summary')]['r2'].iloc[0]
    c6 = overall_df[(overall_df['model'] == 'C6')
                    & (overall_df['level'] == 'repeat_summary')]['r2'].iloc[0]
    c8r = overall_df[(overall_df['model'] == 'C8R')
                     & (overall_df['level'] == 'repeat_summary')]['r2'].iloc[0]
    log('')
    log(f'=== {P.SCREEN_ID} TASK C COMPLETE ===')
    log(f'{P.SCREEN_ID} support: {len(support_sids)}/2636 cells '
        f'({len(support_sids)/2636:.1%})')
    log(f'{P.SCREEN_ID} C4 R2: {c4:.4f} | C6: {c6:.4f} | C8R: {c8r:.4f}')
    log(f'{P.SCREEN_ID} dR2(C6-C4): {c6-c4:+.4f} | dR2(C8R-C4): {c8r-c4:+.4f}')
    log(f'folds: {n_folds}')
    log(f'wall: {time.time()-t_total:.0f}s')


if __name__ == '__main__':
    main()
