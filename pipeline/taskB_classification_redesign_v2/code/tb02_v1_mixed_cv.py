#!/usr/bin/env python3
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, "taskB_classification_redesign_v2")
RES = os.path.join(OUT, "results", "V1_mixed_cv")
FIG = os.path.join(OUT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

N_BOOT = 1000
REPEAT_SEEDS = (42, 43, 44, 45, 46)
RNG_BOOT = np.random.default_rng(20260901)

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

t0 = time.time()
log("=" * 72)
log("Task B Stage 02 — V1 mixed 5x5 repeated CV")
log("=" * 72)

# ────────────────────────────────────────────────────────────── load
X = pd.read_parquet(os.path.join(OUT, "features", "X_all.parquet"))
log(f"X_all: {X.shape}")

A_COLS = [c for c in X.columns if c.startswith("A_")]
B_COLS = [c for c in X.columns if c.startswith("B_")]
C_COLS = [c for c in X.columns if c.startswith("C_")]
D_COLS = [c for c in X.columns if c.startswith("D_")]
assert len(A_COLS) == 24 and len(B_COLS) == 32 and len(C_COLS) == 3 \
    and len(D_COLS) == 17

COMBOS = {
    "M1": {"desc": "D only (confounders)", "cols": D_COLS},
    "M2": {"desc": "A+B+C (communication)", "cols": A_COLS + B_COLS + C_COLS},
    "M3": {"desc": "A+B+C+D (all)",
           "cols": A_COLS + B_COLS + C_COLS + D_COLS},
    "M4": {"desc": "A+D (endpoint+confounders)", "cols": A_COLS + D_COLS},
}
y_all = X["label"].values.astype(np.int64)

# shared (seed, endpoint) pair overlap between datasets — recorded for V2
pairs_ft = set(zip(X.loc[X["dataset"] == "F-TRACT", "seed_roi"],
                   X.loc[X["dataset"] == "F-TRACT", "endpoint_roi"]))
pairs_ner = set(zip(X.loc[X["dataset"] == "NER", "seed_roi"],
                    X.loc[X["dataset"] == "NER", "endpoint_roi"]))
shared_pairs = pairs_ft & pairs_ner
log(f"shared (seed,endpoint) pairs across datasets: {len(shared_pairs)} "
    f"(F-TRACT {len(pairs_ft)}, NER {len(pairs_ner)})")

oof_rows = []
fold_rows = []
sum_rows = []

for mname, spec in COMBOS.items():
    cols = spec["cols"]
    Xm = X[cols].values.astype(np.float64)
    log(f"\n[{mname}] {spec['desc']}: {len(cols)} features")
    model_oof = []          # per-repeat OOF frames
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rep_pred = np.empty(len(X), dtype=np.float64)
        for fold, (tr, te) in enumerate(skf.split(Xm, y_all)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xm[tr])
            Xte = scaler.transform(Xm[te])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf = LogisticRegression(C=1.0, penalty="l2",
                                         max_iter=1000,
                                         class_weight="balanced")
                clf.fit(Xtr, y_all[tr])
                n_warn = sum(1 for x in w
                             if issubclass(x.category, ConvergenceWarning))
            p = clf.predict_proba(Xte)[:, 1]
            rep_pred[te] = p
            # fold metrics
            yt = y_all[te]
            yhat = (p >= 0.5).astype(int)
            auc = roc_auc_score(yt, p)
            acc = accuracy_score(yt, yhat)
            tn, fp, fn, tp = confusion_matrix(yt, yhat).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec_ = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            fold_rows.append({
                "model": mname, "repeat": rep, "seed": seed, "fold": fold,
                "n_test": int(len(te)), "auc": auc, "acc": acc,
                "sens": sens, "spec": spec_, "n_warn": int(n_warn),
            })
        model_oof.append(pd.DataFrame({
            "model": mname, "repeat": rep, "seed": seed,
            "cell_id": X["cell_id"].values,
            "dataset": X["dataset"].values,
            "label": y_all,
            "subject_id": X["subject_id"].values,
            "y_pred": rep_pred,
        }))
    oof = pd.concat(model_oof, ignore_index=True)
    oof_rows.append(oof)

    # ── summary: per-repeat AUC, pooled bootstrap CI (cell-level) ──
    aucs = [roc_auc_score(g["label"].values, g["y_pred"].values)
            for _, g in oof.groupby("repeat")]
    # cell-level bootstrap: resample cells (with all their repeat rows)
    cells = X["cell_id"].values
    labs = X["label"].values
    yp = oof.pivot_table(index="cell_id", columns="repeat",
                         values="y_pred").reindex(cells).values  # (n_cell, 5)
    boot_aucs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([
            RNG_BOOT.choice(np.where(labs == 0)[0],
                            size=int((labs == 0).sum()), replace=True),
            RNG_BOOT.choice(np.where(labs == 1)[0],
                            size=int((labs == 1).sum()), replace=True),
        ])
        yb = labs[idx]
        pb = yp[idx].mean(axis=1)          # cell-level mean over repeats
        boot_aucs[b] = roc_auc_score(yb, pb)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    # pooled metrics
    pooled_auc = roc_auc_score(oof["label"].values, oof["y_pred"].values)
    yhat_p = (oof["y_pred"] >= 0.5).astype(int)
    acc_p = accuracy_score(oof["label"].values, yhat_p)
    tn, fp, fn, tp = confusion_matrix(oof["label"].values, yhat_p).ravel()
    sum_rows.append({
        "model": mname, "n_features": len(cols),
        "desc": spec["desc"],
        "mean_auc": float(np.mean(aucs)), "sd_auc": float(np.std(aucs)),
        "per_repeat_aucs": json.dumps([float(a) for a in aucs]),
        "pooled_auc": float(pooled_auc),
        "auc_ci_lo": float(ci_lo), "auc_ci_hi": float(ci_hi),
        "acc": float(acc_p),
        "sens": float(tp / (tp + fn)), "spec": float(tn / (tn + fp)),
    })
    log(f"  per-repeat AUCs: {[f'{a:.4f}' for a in aucs]}")
    log(f"  mean AUC = {np.mean(aucs):.4f} (sd {np.std(aucs):.4f}), "
        f"bootstrap 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    log(f"  pooled acc={acc_p:.4f} sens={tp/(tp+fn):.4f} "
        f"spec={tn/(tn+fp):.4f}")

# ── M3 - M1 paired delta ──
oof_all = pd.concat(oof_rows, ignore_index=True)
m3 = oof_all[oof_all["model"] == "M3"]
m1 = oof_all[oof_all["model"] == "M1"]
yp3 = m3.pivot_table(index="cell_id", columns="repeat",
                     values="y_pred").reindex(cells).values
yp1 = m1.pivot_table(index="cell_id", columns="repeat",
                     values="y_pred").reindex(cells).values
delta_aucs = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = np.concatenate([
        RNG_BOOT.choice(np.where(labs == 0)[0],
                        size=int((labs == 0).sum()), replace=True),
        RNG_BOOT.choice(np.where(labs == 1)[0],
                        size=int((labs == 1).sum()), replace=True),
    ])
    yb = labs[idx]
    delta_aucs[b] = (roc_auc_score(yb, yp3[idx].mean(axis=1))
                     - roc_auc_score(yb, yp1[idx].mean(axis=1)))
d_lo, d_hi = np.percentile(delta_aucs, [2.5, 97.5])
delta_row = {
    "model": "M3-M1", "n_features": np.nan,
    "desc": "increment: all features minus confounders-only",
    "mean_auc": float(delta_aucs.mean()),
    "sd_auc": float(delta_aucs.std()),
    "per_repeat_aucs": json.dumps([]),
    "pooled_auc": float(delta_aucs.mean()),
    "auc_ci_lo": float(d_lo), "auc_ci_hi": float(d_hi),
    "acc": np.nan, "sens": np.nan, "spec": np.nan,
}
sum_rows.append(delta_row)
log(f"\nM3-M1 delta AUC = {delta_aucs.mean():+.4f}, "
    f"95% CI [{d_lo:+.4f}, {d_hi:+.4f}]")

# ────────────────────────────────────────────────────────────── outputs
oof_all.to_parquet(os.path.join(RES, "V1_OOF_predictions.parquet"),
                   index=False)
pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "V1_metrics_by_fold.csv"),
                               index=False)
pd.DataFrame(sum_rows).to_csv(os.path.join(RES, "V1_summary.csv"),
                              index=False)
log(f"\nsaved: OOF ({oof_all.shape}), metrics_by_fold "
    f"({len(fold_rows)} rows), summary ({len(sum_rows)} rows)")

meta = {
    "n_cells": int(len(X)),
    "n_ftract": int((X["dataset"] == "F-TRACT").sum()),
    "n_ner": int((X["dataset"] == "NER").sum()),
    "n_ner_subjects": int(X["subject_id"].nunique()),
    "shared_seed_endpoint_pairs": int(len(shared_pairs)),
    "cv": "StratifiedKFold(5, shuffle, seeds 42-46)",
    "classifier": "LogisticRegression(C=1.0, l2, max_iter=1000, "
                  "class_weight=balanced)",
    "preprocessing": "StandardScaler fit on training fold only",
    "bootstrap": "cell-level, stratified by label, 1000 resamples",
    "note": "V1 is deliberately naive: no subject grouping and no "
            "(seed,endpoint) overlap control — those are V2 confounder "
            "controls; V1 reproduces the old paradigm",
}
with open(os.path.join(RES, "V1_META.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(RES, "V1_LOG.txt"), "w") as f:
    f.write("\n".join(LOG))
log(f"\nDONE in {time.time()-t0:.0f}s")
