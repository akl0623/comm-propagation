#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, "taskB_classification_redesign_v2")
TBL = os.path.join(OUT, "tables")
os.makedirs(TBL, exist_ok=True)

V1S = os.path.join(OUT, "results", "V1_mixed_cv", "V1_summary.csv")
V2S = os.path.join(OUT, "results", "V2_lodo", "V2_summary.csv")
V2D = os.path.join(OUT, "results", "V2_lodo", "V2_delta_vs_V1.csv")
V2A = os.path.join(OUT, "results", "V2_lodo", "V2_ablation_metrics.csv")
V3S = os.path.join(OUT, "results", "V3_matched", "V3_summary.csv")
V3A = os.path.join(OUT, "results", "V3_matched", "V3_ablation_metrics.csv")
V3M = os.path.join(OUT, "results", "V3_matched", "V3_SMD.csv")
V3Z = os.path.join(OUT, "results", "V3_matched",
                   "V3_matched_sample_sizes.csv")
V4I = os.path.join(OUT, "results", "V4_confound", "V4_increments.csv")
V4F = os.path.join(OUT, "results", "V4_confound", "V4_feature_importance.csv")

v1 = pd.read_csv(V1S)
v2 = pd.read_csv(V2S)
v2d = pd.read_csv(V2D)
v2a = pd.read_csv(V2A)
v3 = pd.read_csv(V3S)
v3a = pd.read_csv(V3A)
v3smd = pd.read_csv(V3M)
v3sz = pd.read_csv(V3Z)
v4i = pd.read_csv(V4I)
v4f = pd.read_csv(V4F)

# ═════════════════════ TABLE 01: all validations x combos ═════════════
rows = []
for m, r in v1.set_index("model").iterrows():
    rows.append({"validation": "V1 mixed 5x5 CV", "direction": "mixed",
                 "model": m, "feature_combo": r["desc"],
                 "n_cells": 3294,
                 "mean_auc": r["mean_auc"], "sd_auc": r["sd_auc"],
                 "ci_lo": r["auc_ci_lo"], "ci_hi": r["auc_ci_hi"],
                 "acc": r["acc"], "sens": r["sens"], "spec": r["spec"],
                 "source": V1S})
for _, r in v2.iterrows():
    rows.append({"validation": f"V2 LODO ({r['variant']})",
                 "direction": r["direction"], "model": r["model"],
                 "feature_combo": "M2 (A+B+C)" if r["model"] == "M2"
                 else "M3 (A+B+C+D)",
                 "n_cells": "see V2_metrics.csv", "mean_auc": r["mean_auc"],
                 "sd_auc": r["sd_auc"], "ci_lo": r["auc_ci_lo"],
                 "ci_hi": r["auc_ci_hi"], "acc": r["mean_acc"],
                 "sens": r["mean_sens"], "spec": r["mean_spec"],
                 "source": V2S})
for _, r in v3.iterrows():
    rows.append({"validation": "V3 matched 5x5 CV", "direction": "matched",
                 "model": r["model"],
                 "feature_combo": "matched: seed-Yeo7 x L-bin, 946 cells",
                 "n_cells": int(r["n_matched_ftract"] + r["n_matched_ner"]),
                 "mean_auc": r["mean_auc"], "sd_auc": r["sd_auc"],
                 "ci_lo": r["auc_ci_lo"], "ci_hi": r["auc_ci_hi"],
                 "acc": r["pooled_acc"], "sens": np.nan, "spec": np.nan,
                 "source": V3S})
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(TBL, "TABLE_01_ALL_RESULTS.csv"), index=False)
print("TABLE_01:", t1.shape)

# ═════════════════════ TABLE 02: LODO comparison + ablation ═══════════
rows = []
for _, r in v2d.iterrows():
    base = v2[(v2["direction"] == r["direction"]) &
              (v2["variant"] == r["variant"]) &
              (v2["model"] == r["model"])].iloc[0]
    rows.append({"section": "lodo_comparison", "direction": r["direction"],
                 "variant": r["variant"], "model": r["model"],
                 "v1_auc": r["V1_auc"], "v2_mean_auc": r["V2_auc"],
                 "v2_sd_auc": base["sd_auc"],
                 "v2_ci_lo": base["auc_ci_lo"],
                 "v2_ci_hi": base["auc_ci_hi"],
                 "delta_auc": r["delta_auc"], "delta_ci_lo": r["delta_ci_lo"],
                 "delta_ci_hi": r["delta_ci_hi"],
                 "source": V2D})
for _, r in v2a.iterrows():
    rows.append({"section": "lodo_ablation", "direction": r["direction"],
                 "variant": "OOS", "model": r["model"],
                 "v1_auc": np.nan, "v2_mean_auc": r["mean_auc"],
                 "v2_sd_auc": r["sd_auc"], "v2_ci_lo": np.nan,
                 "v2_ci_hi": np.nan, "delta_auc": np.nan,
                 "delta_ci_lo": np.nan, "delta_ci_hi": np.nan,
                 "source": V2A})
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(TBL, "TABLE_02_LODO_COMPARISON.csv"), index=False)
print("TABLE_02:", t2.shape)

# ═════════════════════ TABLE 03: matched comparison ═══════════════════
rows = []
for _, r in v3.iterrows():
    v1m = v1[v1["model"] == r["model"]].iloc[0]
    rows.append({"section": "matched_cv", "model": r["model"],
                 "v1_auc": float(v1m["mean_auc"]),
                 "v3_mean_auc": r["mean_auc"], "v3_sd_auc": r["sd_auc"],
                 "v3_ci_lo": r["auc_ci_lo"], "v3_ci_hi": r["auc_ci_hi"],
                 "n_matched": int(r["n_matched_ftract"] + r["n_matched_ner"]),
                 "source": V3S})
for _, r in v3a.iterrows():
    rows.append({"section": "matched_ablation", "model": r["model"],
                 "v1_auc": np.nan, "v3_mean_auc": r["mean_auc"],
                 "v3_sd_auc": r["sd_auc"], "v3_ci_lo": np.nan,
                 "v3_ci_hi": np.nan, "n_matched": 946,
                 "source": V3A})
for _, r in v3smd.iterrows():
    rows.append({"section": "smd", "model": r["variable"],
                 "v1_auc": r["smd"] if r["phase"] == "before" else np.nan,
                 "v3_mean_auc": r["smd"] if r["phase"] == "after" else np.nan,
                 "v3_sd_auc": np.nan, "v3_ci_lo": np.nan,
                 "v3_ci_hi": np.nan, "n_matched": np.nan,
                 "source": V3M})
t3 = pd.DataFrame(rows)
t3.to_csv(os.path.join(TBL, "TABLE_03_MATCHED_COMPARISON.csv"),
          index=False)
print("TABLE_03:", t3.shape)
print("  +n_paths sensitivity (42 cells, non-viable):", V3Z)

# ═════════════════════ TABLE 04: feature importance ═══════════════════
t4 = v4f.copy()
t4.insert(0, "rank", np.arange(1, len(t4) + 1))
t4["source"] = V4F
t4.to_csv(os.path.join(TBL, "TABLE_04_FEATURE_IMPORTANCE.csv"), index=False)
print("TABLE_04:", t4.shape)

# ═════════════════════ TABLE 05: confound decomposition ═══════════════
rows = []
for m, lab in (("M1", "D only (confounders)"), ("M4", "A+D"),
               ("M3", "A+B+C+D (all)")):
    r = v1[v1["model"] == m].iloc[0]
    rows.append({"component": m, "features": lab,
                 "mean_auc": float(r["mean_auc"]),
                 "ci_lo": float(r["auc_ci_lo"]), "ci_hi": float(r["auc_ci_hi"]),
                 "note": "mixed 5x5 CV", "source": V1S})
for _, r in v4i.iterrows():
    rows.append({"component": r["increment"], "features": r["description"],
                 "mean_auc": r["mean_delta"], "ci_lo": r["ci_lo"],
                 "ci_hi": r["ci_hi"],
                 "note": f"paired bootstrap p(<=0)={r['bootstrap_p_le0']:.4f}; "
                         "ceiling: max possible increment = 1 - 0.9913",
                 "source": V4I})
t5 = pd.DataFrame(rows)
t5.to_csv(os.path.join(TBL, "TABLE_05_CONFOUND_DECOMPOSITION.csv"),
          index=False)
print("TABLE_05:", t5.shape)
print("\nall tables written to", TBL)
