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
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, "taskB_classification_redesign_v2")
RES = os.path.join(OUT, "results", "V2_lodo")
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
log("Task B Stage 03 — V2 LODO")
log("=" * 72)

X = pd.read_parquet(os.path.join(OUT, "features", "X_all.parquet"))
A_COLS = [c for c in X.columns if c.startswith("A_")]
B_COLS = [c for c in X.columns if c.startswith("B_")]
C_COLS = [c for c in X.columns if c.startswith("C_")]
D_COLS = [c for c in X.columns if c.startswith("D_")]
COMBOS = {
    "M2": A_COLS + B_COLS + C_COLS,
    "M3": A_COLS + B_COLS + C_COLS + D_COLS,
}

is_ft = (X["dataset"] == "F-TRACT").values
is_ner = ~is_ft
y = X["label"].values.astype(np.int64)
cell_ids = X["cell_id"].values
datasets = X["dataset"].values

# direction spec: (A = train-majority dataset, B = test-majority dataset)
DIRECTIONS = {
    "F_to_N": {"A": is_ft, "B": is_ner, "A_name": "F-TRACT", "B_name": "NER"},
    "N_to_F": {"A": is_ner, "B": is_ft, "A_name": "NER", "B_name": "F-TRACT"},
}
VARIANTS = ("OOS", "LETTER")

def fit_predict(Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                 class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        n_warn = sum(1 for x in w
                     if issubclass(x.category, ConvergenceWarning))
    return clf.predict_proba(Xte_s)[:, 1], n_warn

oof_rows = []
metric_rows = []
sum_rows = []

for dname, d in DIRECTIONS.items():
    A_mask = d["A"]
    B_mask = d["B"]
    nA, nB = int(A_mask.sum()), int(B_mask.sum())
    log(f"\n== {dname}: A={d['A_name']} (n={nA}) -> test on "
        f"{d['B_name']} (n={nB}) ==")
    for variant in VARIANTS:
        for mname, cols in COMBOS.items():
            Xm = X[cols].values.astype(np.float64)
            rep_aucs, rep_accs, rep_sens, rep_spec = [], [], [], []
            var_rows = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                a_idx = np.where(A_mask)[0]
                b_idx = np.where(B_mask)[0]
                a_perm = rng.permutation(a_idx)
                b_perm = rng.permutation(b_idx)
                b_half = len(b_idx) // 2
                b_tr = b_perm[:b_half]
                b_te = b_perm[b_half:]
                if variant == "OOS":
                    a_tr_frac = 0.8
                    a_tr = a_perm[:int(round(0.8 * len(a_idx)))]
                    a_te = a_perm[int(round(0.8 * len(a_idx))):]
                else:  # LETTER
                    a_tr = a_perm
                    a_te = np.array([], dtype=np.int64)
                tr = np.concatenate([a_tr, b_tr])
                te = np.concatenate([a_te, b_te])
                p_tr, n_warn = fit_predict(Xm[tr], y[tr], Xm[tr])
                p_te, _ = fit_predict(Xm[tr], y[tr], Xm[te])
                # rows (roles: train_A / train_B are in-sample, test unseen)
                for i, p, role in ((a_tr, p_tr[:len(a_tr)], "train_A"),
                                   (b_tr, p_tr[len(a_tr):], "train_B"),
                                   (a_te, p_te[:len(a_te)], "test"),
                                   (b_te, p_te[len(a_te):], "test")):
                    if len(i) == 0:
                        continue
                    var_rows.append(pd.DataFrame({
                        "direction": dname, "variant": variant, "model": mname,
                        "repeat": seed,
                        "cell_id": cell_ids[i], "dataset": datasets[i],
                        "label": y[i], "split_role": role, "y_pred": p,
                    }))
                # metrics
                if variant == "OOS":
                    yp = p_te
                    yt = y[te]
                else:  # LETTER: pooled (train-A in-sample + test-B unseen)
                    yp = np.concatenate([p_tr[:len(a_tr)], p_te])
                    yt = np.concatenate([y[a_tr], y[te]])
                auc = roc_auc_score(yt, yp)
                acc = accuracy_score(yt, (yp >= 0.5).astype(int))
                tn, fp, fn, tp = confusion_matrix(
                    yt, (yp >= 0.5).astype(int)).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                rep_aucs.append(auc)
                rep_accs.append(acc)
                rep_sens.append(sens)
                rep_spec.append(spec)
                metric_rows.append({
                    "direction": dname, "variant": variant, "model": mname,
                    "repeat": seed, "n_train": int(len(tr)),
                    "n_train_A": int(len(a_tr)), "n_train_B": int(len(b_tr)),
                    "n_test": int(len(te)), "n_test_A": int(len(a_te)),
                    "n_test_B": int(len(b_te)),
                    "auc": float(auc), "acc": float(acc),
                    "sens": float(sens), "spec": float(spec),
                    "n_warn": int(n_warn),
                })
            vr = pd.concat(var_rows, ignore_index=True)
            vr["_run"] = f"{dname}|{variant}|{mname}"
            oof_rows.append(vr)
            # bootstrap CI on pooled predictions (cell-level, label-strat)
            if variant == "OOS":
                pooled = vr[vr["split_role"] == "test"]
            else:  # LETTER: train_A (in-sample) + test (unseen)
                pooled = vr[vr["split_role"].isin(["train_A", "test"])]
            cells_p = pooled["cell_id"].unique()
            labs_p = pooled.groupby("cell_id")["label"].first().reindex(
                cells_p).values
            pred_p = pooled.groupby("cell_id")["y_pred"].mean().reindex(
                cells_p).values
            boot = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = np.concatenate([
                    RNG_BOOT.choice(np.where(labs_p == 0)[0],
                                    size=int((labs_p == 0).sum()),
                                    replace=True),
                    RNG_BOOT.choice(np.where(labs_p == 1)[0],
                                    size=int((labs_p == 1).sum()),
                                    replace=True),
                ])
                boot[b] = roc_auc_score(labs_p[idx], pred_p[idx])
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            aucs = np.array(rep_aucs)
            sum_rows.append({
                "direction": dname, "variant": variant, "model": mname,
                "mean_auc": float(aucs.mean()),
                "sd_auc": float(aucs.std()),
                "per_repeat_aucs": json.dumps([float(a) for a in aucs]),
                "auc_ci_lo": float(ci_lo), "auc_ci_hi": float(ci_hi),
                "mean_acc": float(np.mean(rep_accs)),
                "mean_sens": float(np.nanmean(rep_sens)),
                "mean_spec": float(np.nanmean(rep_spec)),
                "pooling_note": ("test cells only (out-of-sample)"
                                 if variant == "OOS"
                                 else "train-A (in-sample) + test-B pooled"),
            })
            log(f"  [{variant}] {mname}: per-repeat AUCs "
                f"{[f'{a:.4f}' for a in aucs]}")
            log(f"           mean={aucs.mean():.4f} sd={aucs.std():.4f} "
                f"CI [{ci_lo:.4f}, {ci_hi:.4f}] "
                f"acc={np.mean(rep_accs):.4f} "
                f"sens={np.nanmean(rep_sens):.4f} "
                f"spec={np.nanmean(rep_spec):.4f}")

# ── delta vs V1 (independent-difference bootstrap) ──
v1 = pd.read_csv(os.path.join(OUT, "results", "V1_mixed_cv",
                              "V1_summary.csv"))
v1_auc = dict(zip(v1["model"], v1["mean_auc"]))
oof_all = pd.concat(oof_rows, ignore_index=True)
v1_oof = pd.read_parquet(os.path.join(OUT, "results", "V1_mixed_cv",
                                      "V1_OOF_predictions.parquet"))
X["_order"] = np.arange(len(X))
v1_cells = X["cell_id"].values
v1_labs = X["label"].values

delta_rows = []
for _, r in pd.DataFrame(sum_rows).iterrows():
    m = r["model"]
    v1_auc_m = v1_auc[m]
    v2_auc_m = r["mean_auc"]
    # independent bootstrap: resample V1 cells and V2 pooled cells separately
    sub = oof_all[(oof_all["direction"] == r["direction"]) &
                  (oof_all["variant"] == r["variant"]) &
                  (oof_all["model"] == m)]
    if r["variant"] == "OOS":
        pooled = sub[sub["split_role"] == "test"]
    else:
        pooled = sub[sub["split_role"].isin(["train_A", "test"])]
    cells_p = pooled["cell_id"].unique()
    labs_p = pooled.groupby("cell_id")["label"].first().reindex(
        cells_p).values
    pred_p = pooled.groupby("cell_id")["y_pred"].mean().reindex(
        cells_p).values
    v1p = v1_oof[v1_oof["model"] == m]
    v1p = v1p.groupby("cell_id")["y_pred"].mean().reindex(v1_cells).values
    d_boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i1 = np.concatenate([
            RNG_BOOT.choice(np.where(v1_labs == 0)[0],
                            size=int((v1_labs == 0).sum()), replace=True),
            RNG_BOOT.choice(np.where(v1_labs == 1)[0],
                            size=int((v1_labs == 1).sum()), replace=True),
        ])
        i2 = np.concatenate([
            RNG_BOOT.choice(np.where(labs_p == 0)[0],
                            size=int((labs_p == 0).sum()), replace=True),
            RNG_BOOT.choice(np.where(labs_p == 1)[0],
                            size=int((labs_p == 1).sum()), replace=True),
        ])
        d_boot[b] = (roc_auc_score(v1_labs[i1], v1p[i1])
                     - roc_auc_score(labs_p[i2], pred_p[i2]))
    d_lo, d_hi = np.percentile(d_boot, [2.5, 97.5])
    delta_rows.append({
        "direction": r["direction"], "variant": r["variant"],
        "model": m, "V1_auc": float(v1_auc_m), "V2_auc": float(v2_auc_m),
        "delta_auc": float(v2_auc_m - v1_auc_m),
        "delta_ci_lo": float(d_lo), "delta_ci_hi": float(d_hi),
    })
    log(f"  delta {m} {r['direction']} [{r['variant']}]: "
        f"V1 {v1_auc_m:.4f} -> V2 {v2_auc_m:.4f} "
        f"delta {v2_auc_m - v1_auc_m:+.4f} "
        f"CI [{d_lo:+.4f}, {d_hi:+.4f}]")

# ────────────────────────────────────────────────────────────── outputs
oof_all.to_parquet(os.path.join(RES, "V2_OOF_predictions.parquet"),
                   index=False)
pd.DataFrame(metric_rows).to_csv(os.path.join(RES, "V2_metrics.csv"),
                                 index=False)
pd.DataFrame(sum_rows).to_csv(os.path.join(RES, "V2_summary.csv"),
                              index=False)
pd.DataFrame(delta_rows).to_csv(os.path.join(RES, "V2_delta_vs_V1.csv"),
                                index=False)
log(f"\nsaved: OOF ({oof_all.shape}), metrics ({len(metric_rows)}), "
    f"summary ({len(sum_rows)}), delta ({len(delta_rows)})")

meta = {
    "design": "LODO two variants (spec inconsistency resolved)",
    "OOS": "primary: train=80% A + 50% B; test=20% A + 50% B "
           "(fully out-of-sample, both labels)",
    "LETTER": "spec-literal: train=100% A + 50% B; test=remaining B; "
              "AUC on pooled train-A (in-sample, flagged) + test-B",
    "why_deviation": "literal spec test set is single-label -> AUC undefined",
    "seeds": list(SEEDS),
    "classifier": "LogisticRegression(C=1.0, l2, max_iter=1000, "
                  "class_weight=balanced)",
    "preprocessing": "StandardScaler fit on training rows only",
    "no_subject_grouping": "splits are cell-level random; subject-level "
                           "LODO is deferred to V3",
    "bootstrap": "cell-level, label-stratified, 1000 resamples",
}
with open(os.path.join(RES, "V2_META.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(RES, "V2_LOG.txt"), "w") as f:
    f.write("\n".join(LOG))
log(f"\nDONE in {time.time()-t0:.0f}s")
