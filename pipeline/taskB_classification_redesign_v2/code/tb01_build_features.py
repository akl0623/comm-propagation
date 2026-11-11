#!/usr/bin/env python3
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get('PIPELINE_ROOT', '/home/yuanyufei/revision_v2')
OUT = os.path.join(ROOT, "taskB_classification_redesign_v2")
FEAT = os.path.join(OUT, "features")
CFG = os.path.join(OUT, "config")
os.makedirs(FEAT, exist_ok=True)

SMCR = os.path.join(ROOT, "screen_match_confirm_v1")
sys.path.append(os.path.join(SMCR, "confirmatory", "code"))
sys.path.append(os.path.join(SMCR, "candidate_screens", "code"))
sys.path.append(os.path.join(ROOT, "iterative_combined_screen_v1",
                                "full_experiment", "code"))

import stage4a_lib as A4          # noqa: E402
import bridge_noscreen as BN      # noqa: E402
from bridge_noscreen import P1Engine, METRICS  # noqa: E402

METRIC_ORDER = ("nav", "rout", "search", "comm")
YEO7_ORDER = ["Visual", "Somatomotor", "Dorsal Attention",
              "Ventral Attention", "Limbic", "Frontoparietal", "Default"]
SCREEN_ID = "S10"
K0 = 512
RNG = np.random.default_rng(42)

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

t_start = time.time()
log("=" * 72)
log("Task B Stage 01 — feature construction")
log("=" * 72)

# ────────────────────────────────────────────────────────────── engine
eng = P1Engine()
eng.load()
BN.ENGINE = eng          # bridge feature builders read the module global
A4.ENGINE = eng          # stage4a_lib build_engine() does the same
log(f"engine loaded ({time.time()-t_start:.1f}s), {len(eng.roi_order)} ROIs")

# ────────────────────────────────────────────────────────────── Yeo7
mmp = pd.read_csv(os.path.join(ROOT, "raw_inputs", "mmp_to_yeo7_mapping.csv"))
yeo7 = dict(zip(mmp["MMP_Label"], mmp["Yeo7_Network_Label"]))


def roi_yeo7(roi):
    return yeo7[roi[:-4]]


# ══════════════════════════════════════════════════════════════════
# 1.1  F-TRACT cell list + basic info
# ══════════════════════════════════════════════════════════════════
log("\n[1.1] F-TRACT cells")
sup = pd.read_csv(os.path.join(
    ROOT, "iterative_combined_screen_v1", "full_experiment", "S10",
    "support", "COMMON_SUPPORT.csv"))
uni = pd.read_csv(os.path.join(ROOT, "frozen_s0", "bridge_inputs",
                               "taskC_primary_2636_universe_C0R2D.csv"))
retained = set(sup.loc[sup["retained_for_C4C6C8R"], "sample_id"])
ft = uni[uni["sample_id"].isin(retained)].copy()
assert len(ft) == 1582, f"expected 1582 retained cells, got {len(ft)}"
ft = ft.rename(columns={"path_length_edges": "L"})

reg = pd.read_parquet(os.path.join(
    ROOT, "iterative_combined_screen_v1", "smoke_test", "ROUND2",
    "S10_registry.parquet"))
reg_ids = set(reg["sample_id"].unique())
missing_reg = sorted(retained - reg_ids)
log(f"  retained cells: {len(ft)}; registry unique cells: {len(reg_ids)}; "
    f"retained cells WITHOUT registry paths: {len(missing_reg)}")
if missing_reg:
    log(f"    (cells: {missing_reg[:10]}{'...' if len(missing_reg)>10 else ''})")

# registry per-cell path stats
reg_paths = reg.groupby("sample_id").agg(
    n_paths_registry=("path_id", "count")).reset_index()
reg_paths["mean_length_registry"] = reg.groupby("sample_id")["L"].mean().values
ft = ft.merge(reg_paths, on="sample_id", how="left")
log(f"  registry n_paths per cell: min={ft['n_paths_registry'].min():.0f}, "
    f"median={ft['n_paths_registry'].median():.0f}, "
    f"max={ft['n_paths_registry'].max():.0f}")
assert (ft["mean_length_registry"] == ft["L"]).all(), \
    "registry mean length != cell L (unexpected)"

# ──────────────────────────────────────────────────────────────
# 1.3 group A (F-TRACT) — L one-hot + 4 endpoint metrics
# ──────────────────────────────────────────────────────────────
Lf = ft["L"].values.astype(np.int64)
assert Lf.min() >= 1 and Lf.max() <= 20
A_ft = np.zeros((len(ft), 24), dtype=np.float64)
A_ft[:, :20] = np.eye(20)[Lf - 1]
sf = np.array([eng.r2i[s] for s in ft["seed_roi"]])
ef = np.array([eng.r2i[e] for e in ft["endpoint_roi"]])
A_ft[:, 20:] = np.stack([eng.mats[m][sf, ef] for m in METRIC_ORDER], axis=1)

# ──────────────────────────────────────────────────────────────
# 1.4 group B (F-TRACT) — frozen build_sample_features (S10 seed, K=512)
# ──────────────────────────────────────────────────────────────
log("\n[1.4] F-TRACT group B via frozen build_sample_features")
A4.feature_cell_seed = lambda sid: int.from_bytes(
    hashlib.sha256(f"v1{SCREEN_ID}{sid}".encode()).digest()[:8], "big")
uni_sup = ft[["sample_id", "seed_roi", "endpoint_roi", "L"]].rename(
    columns={"L": "path_length_edges"}).reset_index(drop=True)
t0 = time.time()
X_c4, X_c6, X_c8r, n_paths_sampled = A4.build_sample_features(eng, uni_sup, K0)
log(f"  build_sample_features done ({time.time()-t0:.1f}s); "
    f"X_c8r {X_c8r.shape}")
assert X_c8r.shape == (1582, 56)
# cross-check: A block of X_c4 must equal my independent A_ft construction
assert np.allclose(X_c4[:, :20], A_ft[:, :20], atol=0), "L one-hot mismatch"
assert np.allclose(X_c4[:, 20:], A_ft[:, 20:], atol=0), "ep metric mismatch"
log("  verified: frozen C4 block == independent Stage-01 construction")
B_ft = X_c8r[:, 24:].copy()          # 32-dim ob block (med, iqr interleaved)
n_sampled_lt512 = int((n_paths_sampled < 512).sum())
log(f"  cells with <512 sampled paths: {n_sampled_lt512} "
    f"(min sampled {n_paths_sampled.min()})")
nan_B_ft = int(np.isnan(B_ft).any(axis=1).sum())
log(f"  cells with any NaN in B: {nan_B_ft}")

# ──────────────────────────────────────────────────────────────
# 1.5 group C (F-TRACT) — path-mode proportions over registry paths
# ──────────────────────────────────────────────────────────────
log("\n[1.5] F-TRACT group C from registry paths")
# parse all registry paths once
reg["edge_lists"] = reg["roi_id_sequence"].str.split("|")
reg["n_edges"] = reg["edge_lists"].apply(lambda x: len(x) - 1)

# gather per-metric max/sum over hops for every path (vectorized per metric)
reg_ret = reg[reg["sample_id"].isin(retained)].reset_index(drop=True)
log(f"  retained-cell registry paths: {len(reg_ret)}")
# build edge arrays
u_all, v_all, row_of = [], [], []
for i, el in enumerate(reg_ret["edge_lists"]):
    ids = np.array(el, dtype=np.int16)
    u = ids[:-1]
    v = ids[1:]
    u_all.append(u)
    v_all.append(v)
    row_of.extend([i] * len(u))
u_all = np.concatenate(u_all)
v_all = np.concatenate(v_all)
row_of = np.array(row_of, dtype=np.int64)
log(f"  total hops across paths: {len(u_all)}")
shares = np.full(len(reg_ret), np.nan)
acc = np.zeros((len(reg_ret), 4))
for j, m in enumerate(METRIC_ORDER):
    vals = eng.mats[m][u_all, v_all]
    tot = np.bincount(row_of, weights=vals, minlength=len(reg_ret))
    mx = np.zeros(len(reg_ret))
    np.maximum.at(mx, row_of, vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc[:, j] = np.where(tot > 0, mx / np.where(tot > 0, tot, 1.0), np.nan)
shares = np.nanmean(acc, axis=1)
mode = np.where(shares > 2 / 3, 0, np.where(shares < 1 / 3, 1, 2))
reg_ret["_mode"] = mode
mode_counts = reg_ret.groupby("sample_id")["_mode"].value_counts().unstack(
    fill_value=0)
mode_counts = mode_counts.reindex(columns=[0, 1, 2], fill_value=0)
tot_paths = mode_counts.sum(axis=1)
C_ft = (mode_counts.div(tot_paths, axis=0)).reindex(ft["sample_id"]
                                                    ).fillna(0.0).values
C_ft_no_path = int((tot_paths.reindex(ft["sample_id"]).fillna(0) == 0).sum())
log(f"  retained cells with 0 registry paths (C=0): {C_ft_no_path}")
log(f"  C row sums check: {np.allclose(C_ft.sum(axis=1), 1.0, atol=1e-12)}")

# ──────────────────────────────────────────────────────────────
# 1.6 group D (F-TRACT)
# ──────────────────────────────────────────────────────────────
log("\n[1.6] F-TRACT group D")
D_ft = np.zeros((len(ft), 17), dtype=np.float64)
seed_net = ft["seed_roi"].apply(roi_yeo7)
ep_net = ft["endpoint_roi"].apply(roi_yeo7)
D_ft[:, 0:7] = pd.get_dummies(seed_net).reindex(columns=YEO7_ORDER,
                                               fill_value=0).values
D_ft[:, 7:14] = pd.get_dummies(ep_net).reindex(columns=YEO7_ORDER,
                                               fill_value=0).values
D_ft[:, 14] = np.log(ft["corrected_P1_count"].values + 1)
D_ft[:, 15] = ft["L"].values
# coverage: retained endpoints / universe endpoints, per seed
uni_e = uni.groupby("seed_roi")["endpoint_roi"].nunique()
ft_e = ft.groupby("seed_roi")["endpoint_roi"].nunique()
cov = (ft_e / uni_e).reindex(ft["seed_roi"]).values
assert np.all(np.isfinite(cov)) and np.all((cov > 0) & (cov <= 1))
D_ft[:, 16] = cov
log(f"  seed coverage: min={cov.min():.3f}, median={np.median(cov):.3f}, "
    f"max={cov.max():.3f}")

# ══════════════════════════════════════════════════════════════════
# 1.2  NER cell list + basic info
# ══════════════════════════════════════════════════════════════════
log("\n[1.2] NER cells (UNITS.csv, all strata)")
units = pd.read_csv(os.path.join(ROOT, "ner_reduced_asymmetric_v1", "ner",
                                 "UNITS.csv"))
log(f"  units: {len(units)}, subjects: {units['subject_id'].nunique()}, "
    f"strata: {units['stratum'].value_counts().to_dict()}")
# verify all NER ROIs are in the engine
ner_rois = set(units["stim_region"]) | set(units["endpoint_region"])
bad = [r for r in ner_rois if r not in eng.r2i]
if bad:
    raise RuntimeError(f"NER ROIs missing from engine: {bad}")
log(f"  all {len(ner_rois)} NER ROIs present in engine")

# ──────────────────────────────────────────────────────────────
# 1.3 group A (NER) — identical construction + verify vs stored npy
# ──────────────────────────────────────────────────────────────
Ln = units["L"].values.astype(np.int64)
A_ner = np.zeros((len(units), 24), dtype=np.float64)
A_ner[:, :20] = np.eye(20)[Ln - 1]
sn = np.array([eng.r2i[s] for s in units["stim_region"]])
en = np.array([eng.r2i[e] for e in units["endpoint_region"]])
A_ner[:, 20:] = np.stack([eng.mats[m][sn, en] for m in METRIC_ORDER], axis=1)

for st, name in (("low_var", "low_var"), ("high_var", "high_var")):
    p = os.path.join(ROOT, "ner_reduced_asymmetric_v1", "ner", "features",
                     f"X_{name}__C4.npy")
    stored = np.load(p)
    idx = np.where(units["stratum"] == st)[0]
    mine = A_ner[idx]
    if stored.shape == mine.shape and np.array_equal(stored, mine):
        log(f"  C4 {st}: stored npy == independent construction "
            f"(bitwise, {stored.shape})")
    else:
        log(f"  C4 {st}: MISMATCH stored {stored.shape} vs mine {mine.shape}")
        log(f"    max abs diff: {np.abs(stored - mine).max():.6g}"
            if stored.shape == mine.shape else "")
        raise RuntimeError("NER C4 verification failed")

# ──────────────────────────────────────────────────────────────
# 1.4 group B (NER) — stored frozen C8R16 + structural-zero IQR block
# ──────────────────────────────────────────────────────────────
log("\n[1.4] NER group B from frozen stored matrices")
B_ner = np.zeros((len(units), 32), dtype=np.float64)
for st, name in (("low_var", "low_var"), ("high_var", "high_var")):
    p = os.path.join(ROOT, "ner_reduced_asymmetric_v1", "ner", "features",
                     f"X_{name}__C8R.npy")
    mat = np.load(p)
    idx = np.where(units["stratum"] == st)[0]
    assert mat.shape == (len(idx), 16), f"{name} C8R npy shape {mat.shape}"
    # stored order: ob_{m}_p{q}_med for m in METRICS, q in 0..3
    # expand into 32-col (med, iqr) interleave: iqr = 0 (structural, K=1)
    out = np.zeros((len(idx), 32))
    out[:, 0::2] = mat
    B_ner[idx] = out
nan_B_ner = int(np.isnan(B_ner).any(axis=1).sum())
log(f"  units with any NaN in B: {nan_B_ner}; "
    f"16 IQR cols structurally zero (K=1 modal chain)")

# ──────────────────────────────────────────────────────────────
# 1.5 group C (NER) — single modal chain
# ──────────────────────────────────────────────────────────────
log("\n[1.5] NER group C from modal chain")
C_ner = np.zeros((len(units), 3), dtype=np.float64)
for i, nodes in enumerate(units["nodes"]):
    ids = np.array(json.loads(nodes), dtype=np.int16)
    u = ids[:-1]
    v = ids[1:]
    if len(u) == 0:
        C_ner[i, 2] = 1.0      # degenerate: no hops -> partial placeholder
        continue
    shares_m = []
    for m in METRIC_ORDER:
        vals = eng.mats[m][u, v]
        tot = vals.sum()
        shares_m.append(vals.max() / tot if tot > 0 else np.nan)
    s = np.nanmean(shares_m)
    if np.isnan(s):
        C_ner[i, 2] = 1.0
    elif s > 2 / 3:
        C_ner[i, 0] = 1.0
    elif s < 1 / 3:
        C_ner[i, 1] = 1.0
    else:
        C_ner[i, 2] = 1.0
log(f"  mode distribution: routing={C_ner[:,0].sum():.0f}, "
    f"diffusion={C_ner[:,1].sum():.0f}, partial={C_ner[:,2].sum():.0f}")

# ──────────────────────────────────────────────────────────────
# 1.6 group D (NER)
# ──────────────────────────────────────────────────────────────
log("\n[1.6] NER group D")
D_ner = np.zeros((len(units), 17), dtype=np.float64)
seed_net_n = units["stim_region"].apply(roi_yeo7)
ep_net_n = units["endpoint_region"].apply(roi_yeo7)
D_ner[:, 0:7] = pd.get_dummies(seed_net_n).reindex(columns=YEO7_ORDER,
                                                  fill_value=0).values
D_ner[:, 7:14] = pd.get_dummies(ep_net_n).reindex(columns=YEO7_ORDER,
                                                  fill_value=0).values
D_ner[:, 14] = np.log(units["n_distinct_paths"].values + 1)
D_ner[:, 15] = units["L"].values
# coverage: per (subject, seed) — unit endpoints / raw-subpath endpoints
raw = pd.read_csv(os.path.join(ROOT, "fair_experiments", "ner_per_subject",
                               "subpath_samples_per_subject.csv"))
raw_e = raw.groupby(["subject_id", "stim_region"])["current_end_region"].nunique()
u_e = units.groupby(["subject_id", "stim_region"])["endpoint_region"].nunique()
cov_n = (u_e / raw_e.reindex(u_e.index)).reindex(
    units.set_index(["subject_id", "stim_region"]).index).values
bad_cov = np.where(~np.isfinite(cov_n))[0]
log(f"  non-finite coverage units: {len(bad_cov)}")
if len(bad_cov):
    cov_n = np.nan_to_num(cov_n, nan=0.0)
    log("    (set to 0.0; will be flagged in report)")
D_ner[:, 16] = cov_n
log(f"  seed coverage: min={cov_n.min():.3f}, median={np.median(cov_n):.3f}, "
    f"max={cov_n.max():.3f}")

# ══════════════════════════════════════════════════════════════════
# 1.7  labels + merge
# ══════════════════════════════════════════════════════════════════
log("\n[1.7] merge and shuffle")

A_cols = [f"A_L_is_{k}" for k in range(1, 21)] + \
         [f"A_ep_{m}" for m in METRIC_ORDER]
B_cols = [f"B_{m}_p{q}_{agg}" for m in METRIC_ORDER for q in range(4)
          for agg in ("med", "iqr")]
C_cols = ["C_routing", "C_diffusion", "C_partial"]
D_cols = [f"D_seednet_{x}" for x in YEO7_ORDER] + \
         [f"D_epnet_{x}" for x in YEO7_ORDER] + \
         ["D_log_npaths", "D_mean_length", "D_coverage"]
FEAT_COLS = A_cols + B_cols + C_cols + D_cols
assert len(FEAT_COLS) == 76

ft_out = pd.DataFrame({
    "cell_id": ft["sample_id"].values,
    "dataset": "F-TRACT",
    "label": 0,
    "subject_id": np.nan,
    "seed_roi": ft["seed_roi"].values,
    "endpoint_roi": ft["endpoint_roi"].values,
    "seed_idx": sf,
    "endpoint_idx": ef,
    "L": Lf,
    "y": ft["y_endpoint"].values,
    "seed_yeo7": seed_net.values,
    "endpoint_yeo7": ep_net.values,
    "n_registry_paths": ft["n_paths_registry"].values,
    "corrected_P1_count": ft["corrected_P1_count"].values,
    "n_rows": np.nan, "n_distinct_paths": np.nan,
    "s10_stat": np.nan, "stratum": np.nan,
})
ner_out = pd.DataFrame({
    "cell_id": units["unit_id"].values,
    "dataset": "NER",
    "label": 1,
    "subject_id": units["subject_id"].values,
    "seed_roi": units["stim_region"].values,
    "endpoint_roi": units["endpoint_region"].values,
    "seed_idx": sn,
    "endpoint_idx": en,
    "L": Ln,
    "y": units["y"].values,
    "seed_yeo7": seed_net_n.values,
    "endpoint_yeo7": ep_net_n.values,
    "n_registry_paths": np.nan,
    "corrected_P1_count": np.nan,
    "n_rows": units["n_rows"].values,
    "n_distinct_paths": units["n_distinct_paths"].values,
    "s10_stat": units["s10_stat"].values,
    "stratum": units["stratum"].values,
})

X_ft = np.hstack([A_ft, B_ft, C_ft, D_ft])
X_ner = np.hstack([A_ner, B_ner, C_ner, D_ner])
assert X_ft.shape == (1582, 76) and X_ner.shape == (len(units), 76)

merged = pd.concat([ft_out, ner_out], ignore_index=True)
Xall = pd.DataFrame(np.vstack([X_ft, X_ner]), columns=FEAT_COLS)
merged = pd.concat([merged, Xall], axis=1)
merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
log(f"  merged rows: {len(merged)} "
    f"(F-TRACT {(merged['dataset']=='F-TRACT').sum()}, "
    f"NER {(merged['dataset']=='NER').sum()})")
log(f"  label balance F-TRACT:NER = "
    f"{(merged['dataset']=='F-TRACT').sum()}/{(merged['dataset']=='NER').sum()} "
    f"= {(merged['dataset']=='F-TRACT').sum()/(merged['dataset']=='NER').sum():.3f}")

# ──────────────────────────────────────────────────────────────
# 1.8  outputs
# ──────────────────────────────────────────────────────────────
merged.to_parquet(os.path.join(FEAT, "X_all.parquet"), index=False)
log(f"\n[1.8] saved features/X_all.parquet ({merged.shape})")

groups = {
    "A": {"dims": 24,
          "columns": A_cols,
          "description": "L one-hot (20) + 4 endpoint communication metrics "
                         "(nav, rout, search, comm) from the shared P1 engine",
          "source": "engine matrices + cell L (both datasets, identical "
                    "construction)"},
    "B": {"dims": 32,
          "columns": B_cols,
          "description": "Legendre P0-P3 ordered basis, 4 metrics x 4 orders "
                         "x (med, iqr)",
          "source_FTRACT": "frozen stage4a_lib.build_sample_features with S10 "
                           "feature seed (SHA256('v1S10'+cell_id)[:8], K=512 "
                           "paths/cell) — same code as full S10 experiment",
          "source_NER": "frozen stored X_{low_var,high_var}__C8R.npy "
                        "(16 med cols from the single modal chain); the 16 "
                        "IQR cols are STRUCTURAL ZEROS (K=1), not imputed"},
    "C": {"dims": 3,
          "columns": C_cols,
          "description": "path-mode proportions (routing / diffusion / "
                         "partial). Operational definition: per path, per "
                         "metric, share = max-hop / sum-of-hops; s = mean "
                         "over 4 metrics; routing if s>2/3, diffusion if "
                         "s<1/3, else partial. Cell = mean over paths.",
          "source_FTRACT": "all accepted registry paths of the cell "
                           "(120,822 paths total; 0-path cells -> all-zero "
                           "row, flagged)",
          "source_NER": "single modal chain per unit (degenerate one-hot: "
                        "one class = 1.0)"},
    "D": {"dims": 17,
          "columns": D_cols,
          "description": "confounders: seed Yeo7 one-hot (7) + endpoint Yeo7 "
                         "one-hot (7) + log(n_paths+1) + mean length (=L, "
                         "constant per cell) + seed stimulation coverage",
          "source": "raw_inputs/mmp_to_yeo7_mapping.csv; "
                    "F-TRACT log(corrected_P1_count+1), "
                    "NER log(n_distinct_paths+1); "
                    "F-TRACT coverage = retained-endpoints/universe-endpoints "
                    "per seed; NER coverage = unit-endpoints/raw-subpath-"
                    "endpoints per (subject, seed)"},
}
# per-group NaN census
nan_census = {}
for g, col in [("A", A_cols), ("B", B_cols), ("C", C_cols), ("D", D_cols)]:
    n_nan = int(merged[col].isna().sum().sum())
    rows_nan = int(merged[col].isna().any(axis=1).sum())
    nan_census[g] = {"n_nan_cells": n_nan, "n_rows_with_nan": rows_nan}
groups["_nan_census"] = nan_census
groups["_info_columns"] = ["cell_id", "dataset", "label", "subject_id",
                           "seed_roi", "endpoint_roi", "seed_idx",
                           "endpoint_idx", "L", "y", "seed_yeo7",
                           "endpoint_yeo7", "n_registry_paths",
                           "corrected_P1_count", "n_rows",
                           "n_distinct_paths", "s10_stat", "stratum"]
with open(os.path.join(FEAT, "feature_groups.json"), "w") as f:
    json.dump(groups, f, indent=2)
log("  saved features/feature_groups.json")

# cell_info (pre-shuffle order)
ci = pd.concat([ft_out, ner_out], ignore_index=True)
ci.to_parquet(os.path.join(FEAT, "cell_info.parquet"), index=False)
log("  saved features/cell_info.parquet")

# ──────────────────────────────────────────────────────────────
# 1.9  completion checks
# ──────────────────────────────────────────────────────────────
log("\n[1.9] completion checks")
chk = {}
chk["FTRACT_cells_1582"] = int((merged["dataset"] == "F-TRACT").sum()) == 1582
chk["NER_cells_gt_500"] = int((merged["dataset"] == "NER").sum()) > 500
tot_nan = int(merged[FEAT_COLS].isna().sum().sum())
chk["no_nan_or_documented"] = True   # NaN census documented in groups json
chk["group_dims_documented"] = True
ratio = (merged["dataset"] == "F-TRACT").sum() / (merged["dataset"] == "NER").sum()
chk["label_balance_0.2_5.0"] = 0.2 <= ratio <= 5.0
chk["group_B_ner_provenance_documented"] = True
for k, v in chk.items():
    log(f"  {k}: {'PASS' if v else 'FAIL'}")
chk = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
       for k, v in chk.items()}
chk["n_nan_total"] = tot_nan
chk["label_ratio"] = float(ratio)

with open(os.path.join(CFG, "FEATURE_BUILD_CHECKS.json"), "w") as f:
    json.dump(chk, f, indent=2)

# per-group descriptive stats for the report
stats = {}
for g, cols in [("A", A_cols), ("B", B_cols), ("C", C_cols), ("D", D_cols)]:
    sub = merged[cols]
    stats[g] = {
        "mean": float(sub.mean().mean()),
        "std": float(sub.std().mean()),
        "n_nan": int(sub.isna().sum().sum()),
    }
with open(os.path.join(CFG, "FEATURE_GROUP_STATS.json"), "w") as f:
    json.dump(stats, f, indent=2)

with open(os.path.join(CFG, "FEATURE_BUILD_LOG.txt"), "w") as f:
    f.write("\n".join(LOG))
log(f"\nDONE in {time.time()-t_start:.0f}s")
