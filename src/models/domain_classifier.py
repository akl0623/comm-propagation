#!/usr/bin/env python3
"""
tb05_v4_confound.py — Task B Stage 05: V4 confounder contribution +
feature importance.

Part 1 (5.1/5.2): mixed 5x5 CV for M1 (D), M3 (all), M4 (A+D), M2 (A+B+C) —
identical protocol to V1 (deterministic reproducibility check included).
Increments with paired cell-level bootstrap (1000 resamples, label-stratified,
same resample across models):
  M3 - M1  = communication-feature increment (A+B+C over D)
  M4 - M1  = endpoint-feature increment (A over D)
  M3 - M4  = process-feature increment (B+C over A)
Bootstrap p = fraction of resamples with delta <= 0.

Part 2 (5.3): feature importance on M3, full-data fits (interpretation
models, no CV):
  LR   : StandardScaler on all data, LogisticRegression(C=1.0, balanced),
         |coef| of the standardized model.
  RF   : RandomForestClassifier(500 trees, class_weight='balanced'),
         feature_importances_ — sensitivity analysis.
Group shares (A/B/C/D) from both. Top-10 by LR |coef| (primary) with RF
values alongside. Spearman rank correlation LR vs RF reported.

Outputs:
  results/V4_confound/V4_metrics.csv
  results/V4_confound/V4_increments.csv
  results/V4_confound/V4_feature_importance.csv
  results/V4_confound/V4_group_importance.csv
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

ROOT = PIPELINE_ROOT
OUT = os.path.join(ROOT, "taskB_classification_redesign_v2")
RES = os.path.join(OUT, "results", "V4_confound")
os.makedirs(RES, exist_ok=True)

N_BOOT = 1000
SEEDS = (42, 43, 44, 45, 46)
RNG_BOOT = np.random.default_rng(20260901)

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

t0 = time.time()
log("=" * 72)
log("Task B Stage 05 — V4 confounder contribution + feature importance")
log("=" * 72)

X = pd.read_parquet(os.path.join(OUT, "features", "X_all.parquet"))
A_COLS = [c for c in X.columns if c.startswith("A_")]
B_COLS = [c for c in X.columns if c.startswith("B_")]
C_COLS = [c for c in X.columns if c.startswith("C_")]
D_COLS = [c for c in X.columns if c.startswith("D_")]
COMBOS = {
    "M1": D_COLS,
    "M2": A_COLS + B_COLS + C_COLS,
    "M3": A_COLS + B_COLS + C_COLS + D_COLS,
    "M4": A_COLS + D_COLS,
}
y_all = X["label"].values.astype(np.int64)
cells = X["cell_id"].values
labs = X["label"].values

# ─────────────────────────── Part 1: CV (V1 protocol) ─────────────────
fold_rows = []
oof_store = {}
for mname, cols in COMBOS.items():
    Xm = X[cols].values.astype(np.float64)
    rep_oof = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rep_pred = np.empty(len(X))
        for fold, (tr, te) in enumerate(skf.split(Xm, y_all)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xm[tr])
            Xte = scaler.transform(Xm[te])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                         class_weight="balanced")
                clf.fit(Xtr, y_all[tr])
                n_warn = sum(1 for x in w
                             if issubclass(x.category, ConvergenceWarning))
            p = clf.predict_proba(Xte)[:, 1]
            rep_pred[te] = p
            fold_rows.append({
                "model": mname, "repeat": seed, "fold": fold,
                "auc": float(roc_auc_score(y_all[te], p)),
                "acc": float(accuracy_score(
                    y_all[te], (p >= 0.5).astype(int))),
                "n_warn": int(n_warn),
            })
        rep_oof.append(pd.DataFrame({"model": mname, "repeat": seed,
                                     "cell_id": cells, "label": y_all,
                                     "y_pred": rep_pred}))
    oof_store[mname] = pd.concat(rep_oof, ignore_index=True)
    auc_rep = [roc_auc_score(g["label"].values, g["y_pred"].values)
               for _, g in oof_store[mname].groupby("repeat")]
    log(f"  {mname}: per-repeat AUCs {[f'{a:.4f}' for a in auc_rep]}, "
        f"mean {np.mean(auc_rep):.4f}")

# reproducibility check vs V1 summary
v1 = pd.read_csv(os.path.join(OUT, "results", "V1_mixed_cv",
                              "V1_summary.csv")).set_index("model")
for m in ("M1", "M3", "M4"):
    mine = float(np.mean([roc_auc_score(g["label"].values,
                                        g["y_pred"].values)
                          for _, g in oof_store[m].groupby("repeat")]))
    ref = float(v1.loc[m, "mean_auc"])
    assert abs(mine - ref) < 1e-9, f"{m}: V4 {mine} != V1 {ref}"
log("  reproducibility vs V1: identical (M1/M3/M4)")

pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "V4_metrics.csv"),
                               index=False)

# ─────────────────────────── Part 2: paired bootstrap increments ─────
def cell_preds(model):
    return oof_store[model].pivot_table(index="cell_id", columns="repeat",
                                        values="y_pred").reindex(cells).values

yp = {m: cell_preds(m) for m in COMBOS}
# precompute resample index sets (shared across models for pairing)
boot_idx = []
for b in range(N_BOOT):
    boot_idx.append(np.concatenate([
        RNG_BOOT.choice(np.where(labs == 0)[0],
                        size=int((labs == 0).sum()), replace=True),
        RNG_BOOT.choice(np.where(labs == 1)[0],
                        size=int((labs == 1).sum()), replace=True),
    ]))

def auc_boot(model, idx):
    return roc_auc_score(labs[idx], yp[model][idx].mean(axis=1))

inc_spec = [("M3-M1", "M3", "M1", "communication (A+B+C) over confounders (D)"),
            ("M4-M1", "M4", "M1", "endpoint (A) over confounders (D)"),
            ("M3-M4", "M3", "M4", "process (B+C) over endpoint (A)")]
inc_rows = []
for name, hi, lo, desc in inc_spec:
    a_hi = np.array([auc_boot(hi, i) for i in boot_idx])
    a_lo = np.array([auc_boot(lo, i) for i in boot_idx])
    delta = a_hi - a_lo
    d_lo, d_hi = np.percentile(delta, [2.5, 97.5])
    p = float((delta <= 0).mean())
    inc_rows.append({
        "increment": name, "description": desc,
        "mean_delta": float(delta.mean()), "ci_lo": float(d_lo),
        "ci_hi": float(d_hi), "bootstrap_p_le0": p,
        "base_auc": float(np.array([auc_boot(lo, i)
                                    for i in boot_idx]).mean()),
        "ceiling_note": "AUC increments are ceiling-limited: "
                        "max possible = 1 - base_auc",
    })
    log(f"  {name}: delta {delta.mean():+.5f} CI [{d_lo:+.5f}, {d_hi:+.5f}] "
        f"p(<=0)={p:.4f}")
inc = pd.DataFrame(inc_rows)
inc.to_csv(os.path.join(RES, "V4_increments.csv"), index=False)

# ─────────────────────────── Part 3: feature importance (full data) ───
log("\n[5.3] feature importance on M3 (full-data interpretation fits)")
cols_m3 = COMBOS["M3"]
Xm3 = X[cols_m3].values.astype(np.float64)
scaler = StandardScaler().fit(Xm3)
Xz = scaler.transform(Xm3)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr = LogisticRegression(C=1.0, penalty="l2", max_iter=10000,
                            class_weight="balanced")
    lr.fit(Xz, y_all)
lr_imp = np.abs(lr.coef_[0])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Xz, y_all)
rf_imp = rf.feature_importances_

rho, _ = spearmanr(lr_imp, rf_imp)
log(f"  Spearman(LR |coef|, RF importances) = {rho:.3f}")

imp = pd.DataFrame({
    "feature": cols_m3,
    "group": [c.split("_")[0] for c in cols_m3],
    "lr_abs_coef": lr_imp,
    "lr_share_pct": 100 * lr_imp / lr_imp.sum(),
    "rf_importance": rf_imp,
    "rf_share_pct": 100 * rf_imp / rf_imp.sum(),
})
imp = imp.sort_values("lr_abs_coef", ascending=False).reset_index(drop=True)
imp.to_csv(os.path.join(RES, "V4_feature_importance.csv"), index=False)

grp = imp.groupby("group")[["lr_share_pct", "rf_share_pct"]].sum() \
    .reset_index()
grp["n_features"] = imp.groupby("group").size().values
grp = grp.sort_values("lr_share_pct", ascending=False)
grp.to_csv(os.path.join(RES, "V4_group_importance.csv"), index=False)
log("\n  group shares (LR):")
log(grp.to_string(index=False))
log("\n  top-10 by LR |coef|:")
log(imp.head(10)[["feature", "group", "lr_share_pct", "rf_share_pct"]]
    .to_string(index=False))

meta = {
    "cv_protocol": "identical to V1 (5x5 stratified, LogReg C=1.0 l2 "
                   "balanced, StandardScaler train-only); M1/M3/M4 "
                   "reproduced bitwise vs V1 summary",
    "increments": "paired cell-level bootstrap, 1000 resamples, "
                  "label-stratified, same resample across models; "
                  "p = fraction of deltas <= 0",
    "importance_models": "LR: StandardScaler on all 3294 cells (full-data "
                         "interpretation fit), |coef| of standardized "
                         "model. RF: 500 trees, class_weight=balanced, "
                         "random_state=42 — sensitivity only",
    "ceiling_note": "all increments ceiling-limited: with M1 AUC=0.9913 "
                    "the maximum possible increment is +0.0087",
}
with open(os.path.join(RES, "V4_META.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(RES, "V4_LOG.txt"), "w") as f:
    f.write("\n".join(LOG))
log(f"\nDONE in {time.time()-t0:.0f}s")
