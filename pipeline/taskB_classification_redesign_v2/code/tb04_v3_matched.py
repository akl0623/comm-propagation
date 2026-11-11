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
RES = os.path.join(OUT, "results", "V3_matched")
os.makedirs(RES, exist_ok=True)

N_BOOT = 1000
SEEDS = (42, 43, 44, 45, 46)
RNG_BOOT = np.random.default_rng(20260901)
YEO7_ORDER = ["Visual", "Somatomotor", "Dorsal Attention",
              "Ventral Attention", "Limbic", "Frontoparietal", "Default"]

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

t0 = time.time()
log("=" * 72)
log("Task B Stage 04 — V3 matched subsampling")
log("=" * 72)

X = pd.read_parquet(os.path.join(OUT, "features", "X_all.parquet"))
X = X.reset_index(drop=True)
A_COLS = [c for c in X.columns if c.startswith("A_")]
B_COLS = [c for c in X.columns if c.startswith("B_")]
C_COLS = [c for c in X.columns if c.startswith("C_")]
D_COLS = [c for c in X.columns if c.startswith("D_")]
COMBOS = {"M2": A_COLS + B_COLS + C_COLS,
          "M3": A_COLS + B_COLS + C_COLS + D_COLS}

def lbin(L):
    if L == 1:
        return "L=1"
    if L <= 3:
        return "L=2-3"
    if L <= 6:
        return "L=4-6"
    if L <= 10:
        return "L=7-10"
    return "L>10"

def nbin(v):
    if v <= 1:
        return "n=1"
    if v <= 5:
        return "n=2-5"
    if v <= 20:
        return "n=6-20"
    return "n>20"

X["_lbin"] = X["L"].apply(lbin)
# per-dataset n_paths (different definitions, per spec deviation note)
X["_nbin"] = np.where(X["dataset"] == "F-TRACT",
                      X["n_registry_paths"].apply(nbin),
                      X["n_rows"].apply(nbin))

is_ft = (X["dataset"] == "F-TRACT").values
y = X["label"].values.astype(np.int64)

# ─────────────────────────────────────────────────────────── SMD helper
def smd(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.var(a, ddof=1), np.var(b, ddof=1)
    if na + nb == 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt((na + nb) / 2))

def smd_block(df, cols):
    a = df.loc[df["dataset"] == "F-TRACT", cols].values
    b = df.loc[df["dataset"] == "NER", cols].values
    return float(np.max([abs(smd(a[:, j], b[:, j])) for j in range(a.shape[1])]))

def smd_table(df):
    rows = []
    ft = df[df["dataset"] == "F-TRACT"]
    ner = df[df["dataset"] == "NER"]
    rows.append({"variable": "L (continuous)", "smd": smd(ft["L"], ner["L"])})
    net_d = pd.get_dummies(df["seed_yeo7"]).reindex(columns=YEO7_ORDER,
                                                    fill_value=0)
    ft_d = net_d[df["dataset"] == "F-TRACT"]
    ner_d = net_d[df["dataset"] == "NER"]
    rows.append({"variable": "seed network (max |SMD| over 7 dummies)",
                 "smd": float(np.max([abs(smd(ft_d[c], ner_d[c]))
                                      for c in YEO7_ORDER]))})
    ep_d = pd.get_dummies(df["endpoint_yeo7"]).reindex(columns=YEO7_ORDER,
                                                       fill_value=0)
    ft_e = ep_d[df["dataset"] == "F-TRACT"]
    ner_e = ep_d[df["dataset"] == "NER"]
    rows.append({"variable": "endpoint network (max |SMD|, NOT matched)",
                 "smd": float(np.max([abs(smd(ft_e[c], ner_e[c]))
                                      for c in YEO7_ORDER]))})
    rows.append({"variable": "endpoint metrics (max |SMD| over 4)",
                 "smd": smd_block(df, A_COLS[20:])})
    return pd.DataFrame(rows)

smd_before = smd_table(X)
log("SMD before matching:")
log(smd_before.to_string(index=False))

# ─────────────────────────────────────────────────────────── matching
size_rows = []
oof_rows = []
fold_rows = []
sum_rows = []

def run_cv(matched, mname, cols, label_prefix):
    """V1-identical CV on the matched sample; returns (oof, folds, summ)."""
    Xm = matched[cols].values.astype(np.float64)
    ym = matched["label"].values.astype(np.int64)
    oof_parts, folds, aucs = [], [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        rep_pred = np.empty(len(matched))
        for fold, (tr, te) in enumerate(skf.split(Xm, ym)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xm[tr])
            Xte = scaler.transform(Xm[te])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                         class_weight="balanced")
                clf.fit(Xtr, ym[tr])
                n_warn = sum(1 for x in w
                             if issubclass(x.category, ConvergenceWarning))
            p = clf.predict_proba(Xte)[:, 1]
            rep_pred[te] = p
            yt = ym[te]
            yhat = (p >= 0.5).astype(int)
            auc = roc_auc_score(yt, p)
            tn, fp, fn, tp = confusion_matrix(yt, yhat).ravel()
            folds.append({
                "model": mname, "repeat": seed, "fold": fold,
                "n_test": int(len(te)), "auc": float(auc),
                "acc": float(accuracy_score(yt, yhat)),
                "sens": float(tp / (tp + fn)),
                "spec": float(tn / (tn + fp)),
                "n_warn": int(n_warn),
            })
        aucs.append(roc_auc_score(ym, rep_pred))
        oof_parts.append(pd.DataFrame({
            "model": mname, "repeat": seed,
            "cell_id": matched["cell_id"].values,
            "dataset": matched["dataset"].values,
            "label": ym, "y_pred": rep_pred,
        }))
    oof = pd.concat(oof_parts, ignore_index=True)
    # cell-level stratified bootstrap on pooled OOF
    cells_p = matched["cell_id"].values
    labs_p = matched["label"].values
    pred_p = oof.pivot_table(index="cell_id", columns="repeat",
                             values="y_pred").reindex(cells_p).values
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([
            RNG_BOOT.choice(np.where(labs_p == 0)[0],
                            size=int((labs_p == 0).sum()), replace=True),
            RNG_BOOT.choice(np.where(labs_p == 1)[0],
                            size=int((labs_p == 1).sum()), replace=True),
        ])
        boot[b] = roc_auc_score(labs_p[idx], pred_p[idx].mean(axis=1))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    aucs = np.array(aucs)
    summ = {
        "model": mname, "n_matched_ftract": int((matched["dataset"] == "F-TRACT").sum()),
        "n_matched_ner": int((matched["dataset"] == "NER").sum()),
        "mean_auc": float(aucs.mean()), "sd_auc": float(aucs.std()),
        "per_repeat_aucs": json.dumps([float(a) for a in aucs]),
        "auc_ci_lo": float(ci_lo), "auc_ci_hi": float(ci_hi),
        "pooled_acc": float(accuracy_score(
            oof["label"], (oof["y_pred"] >= 0.5).astype(int))),
    }
    return oof, folds, summ

for rep, seed in enumerate(SEEDS):
    rng = np.random.default_rng(seed)
    # ── primary: seed-net x L-bin ──
    ft_i = np.where(is_ft)[0]
    ner_i = np.where(~is_ft)[0]
    keys_ft = set(zip(X.loc[ft_i, "seed_yeo7"], X.loc[ft_i, "_lbin"]))
    keys_ner = set(zip(X.loc[ner_i, "seed_yeo7"], X.loc[ner_i, "_lbin"]))
    common = keys_ft & keys_ner
    pick_ft, pick_ner = [], []
    for k in sorted(common):
        f = ft_i[[(X.loc[i, "seed_yeo7"], X.loc[i, "_lbin"]) == k
                  for i in ft_i]]
        n = ner_i[[(X.loc[i, "seed_yeo7"], X.loc[i, "_lbin"]) == k
                   for i in ner_i]]
        m = min(len(f), len(n))
        pick_ft.extend(rng.choice(f, size=m, replace=False))
        pick_ner.extend(rng.choice(n, size=m, replace=False))
        size_rows.append({
            "repeat": seed, "variant": "primary",
            "seed_network": k[0], "L_bin": k[1],
            "n_ftract_before": int(len(f)), "n_ner_before": int(len(n)),
            "n_matched_each": int(m),
        })
    matched = X.iloc[np.sort(np.concatenate([pick_ft, pick_ner]))]
    matched = matched.reset_index(drop=True)
    log(f"\nrepeat {seed} PRIMARY (seed-net x L-bin): matched "
        f"{len(matched)} cells (F-TRACT "
        f"{(matched['dataset']=='F-TRACT').sum()}, NER "
        f"{(matched['dataset']=='NER').sum()}); strata {len(common)}")
    # ── sensitivity: + n_paths bin ──
    keys_ft3 = set(zip(X.loc[ft_i, "seed_yeo7"], X.loc[ft_i, "_lbin"],
                       X.loc[ft_i, "_nbin"]))
    keys_ner3 = set(zip(X.loc[ner_i, "seed_yeo7"], X.loc[ner_i, "_lbin"],
                        X.loc[ner_i, "_nbin"]))
    common3 = keys_ft3 & keys_ner3
    pick_ft3, pick_ner3 = [], []
    for k in sorted(common3):
        f = ft_i[[(X.loc[i, "seed_yeo7"], X.loc[i, "_lbin"],
                   X.loc[i, "_nbin"]) == k for i in ft_i]]
        n = ner_i[[(X.loc[i, "seed_yeo7"], X.loc[i, "_lbin"],
                    X.loc[i, "_nbin"]) == k for i in ner_i]]
        m = min(len(f), len(n))
        pick_ft3.extend(rng.choice(f, size=m, replace=False))
        pick_ner3.extend(rng.choice(n, size=m, replace=False))
    matched3 = X.iloc[np.sort(np.concatenate([pick_ft3, pick_ner3]))]
    size_rows.append({
        "repeat": seed, "variant": "plus_npaths",
        "seed_network": "ALL", "L_bin": "ALL",
        "n_ftract_before": int(is_ft.sum()),
        "n_ner_before": int((~is_ft).sum()),
        "n_matched_each": int(len(matched3) / 2) if len(matched3) else 0,
    })
    log(f"repeat {seed} +n_paths sensitivity: matched {len(matched3)} "
        f"(strata {len(common3)})"
        + ("  [NON-VIABLE <600 total]" if len(matched3) < 600 else ""))

    # ── CV on primary matched sample ──
    for mname, cols in COMBOS.items():
        oof, folds, summ = run_cv(matched, mname, cols, f"r{rep}")
        oof["match_repeat"] = rep
        oof_rows.append(oof)
        fold_rows.extend(folds)
        sum_rows.append({"repeat": rep, **summ})
        log(f"  {mname}: mean AUC {summ['mean_auc']:.4f} "
            f"(sd {summ['sd_auc']:.4f}) CI [{summ['auc_ci_lo']:.4f}, "
            f"{summ['auc_ci_hi']:.4f}]")

    if rep == 0:
        smd_after = smd_table(matched)
        log("\nSMD after matching (repeat 0):")
        log(smd_after.to_string(index=False))

# ─────────────────────────────────────────────────────────── outputs
sizes = pd.DataFrame(size_rows)
sizes.to_csv(os.path.join(RES, "V3_matched_sample_sizes.csv"), index=False)
pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "V3_metrics.csv"),
                               index=False)
oof_all = pd.concat(oof_rows, ignore_index=True)
oof_all.to_parquet(os.path.join(RES, "V3_OOF_predictions.parquet"),
                   index=False)

# summary: aggregate over match repeats
summ_df = pd.DataFrame(sum_rows)
agg = summ_df.groupby("model").agg(
    n_matched_ftract=("n_matched_ftract", "first"),
    n_matched_ner=("n_matched_ner", "first"),
    mean_auc=("mean_auc", "mean"),
    sd_auc=("mean_auc", "std"),
    auc_ci_lo=("auc_ci_lo", "min"),
    auc_ci_hi=("auc_ci_hi", "max"),
    pooled_acc=("pooled_acc", "mean"),
).reset_index()
agg.to_csv(os.path.join(RES, "V3_summary.csv"), index=False)
log("\nV3 summary:")
log(agg.to_string(index=False))

smd_out = smd_before.copy()
smd_out["phase"] = "before"
smd_after2 = smd_after.copy()
smd_after2["phase"] = "after"
pd.concat([smd_out, smd_after2]).to_csv(os.path.join(RES, "V3_SMD.csv"),
                                        index=False)

meta = {
    "matching_method": "stratified subsampling: per common (seed Yeo7, L-bin) "
                       "stratum draw min(n_F, n_N) from each dataset; "
                       "single-dataset strata excluded; 5 repeats (seeds 42-46)",
    "L_bins": ["L=1", "L=2-3", "L=4-6", "L=7-10", "L>10"],
    "n_paths_deviation": "path-count columns are definitionally incompatible "
                         "across datasets (F-TRACT n_registry_paths screen-"
                         "capped median 122 vs NER n_rows max 4); spec bins "
                         "(1, 2-5, 6-20, >20) empty most strata -> 3-variable "
                         "variant run as sensitivity and reported (expected "
                         "non-viable, itself a domain-difference finding)",
    "endpoint_network_not_matched": "spec matches seed network only; endpoint "
                                    "network SMD reported and flagged as "
                                    "uncontrolled",
    "cv": "identical to V1",
    "classifier": "LogisticRegression(C=1.0, l2, max_iter=1000, "
                  "class_weight=balanced)",
    "bootstrap": "cell-level, label-stratified, 1000 resamples",
}
with open(os.path.join(RES, "V3_META.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(RES, "V3_LOG.txt"), "w") as f:
    f.write("\n".join(LOG))
log(f"\nDONE in {time.time()-t0:.0f}s")
