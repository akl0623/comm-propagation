#!/usr/bin/env python3
"""E2: matched F-TRACT/NER external-domain stress test - DESCRIPTIVE_ONLY mode.

Matched units built on the frozen label-blind candidate path library (L1 pairs)
and common seed-endpoint support. No legacy source-specific path pools.
Primary question: incremental domain-discrimination performance of C+P over the
confound baseline C0. Leave-one-seed-out folds; in-fold pair weights;
within-pair label permutation; seed-cluster bootstrap. Descriptive only.
fit_models(U, C) is reusable for stage 09 sensitivity variants.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

OUT = os.path.join(PIPELINE_ROOT, "08_e2_matched_domain_test")
PART5 = os.path.join(PIPELINE_ROOT, "05_identity_reconstruction", "partitions")
HOME = os.path.join(PIPELINE_ROOT, "raw_inputs")
RNG = np.random.default_rng(42)
N_BOOT = 1000
N_PERM = 200

REGIONS_SRC = os.path.join(PIPELINE_ROOT, "code", "legacy_snapshot", "batch statis from new dataset.py")

MODEL_COLS = {
    "C0": ["n_subjects_supporting_pair", "onset_available", "n_runs"],
    "C1": ["seed_hemi", "end_hemi", "seed_yeo7_id", "end_yeo7_id",
           "tract_length_mm", "length_tercile", "n_subjects_supporting_pair"],
    "P1": ["response_probability", "onset_ms_median", "nav_eff", "rout_eff",
           "search_info", "comm_norm"],
}
MODEL_COLS["C+P"] = MODEL_COLS["C0"] + MODEL_COLS["C1"] + MODEL_COLS["P1"]


def load_region_order():
    import re
    src = open(REGIONS_SRC, encoding="utf-8").read()
    m = re.search(r"regions\s*=\s*\[(.*?)\]", src, re.S)
    return re.findall(r"'([^']+)'", m.group(1))


def load_struct():
    mats = {}
    for k, f in {"tract_length": "averageConnectivity_tractLength_0.25density.csv",
                 "nav_eff": "navigation_efficiency_matrix_0.25density.csv",
                 "rout_eff": "rout_efficiency_matrix_0.25density.csv",
                 "search_info": "search_information_matrix_0.25density.csv",
                 "communicability": "communicability_matrix_0.25density.csv"}.items():
        mats[k] = np.loadtxt(f"{HOME}/{f}", delimiter=",")
    c = mats["communicability"]
    mats["comm_norm"] = (c - np.nanmin(c)) / (np.nanmax(c) - np.nanmin(c))
    return mats


def load_ftract(prob_file="Reordered_matrix_probability.csv",
                onset_file="Reordered_matrix_onset_delay__median.csv"):
    prob = pd.read_csv(f"{HOME}/{prob_file}", index_col=0)
    onset = pd.read_csv(f"{HOME}/{onset_file}", index_col=0)
    return prob, onset


def load_yeo7():
    df = pd.read_csv(f"{HOME}/mmp_to_yeo7_mapping.csv")
    return (dict(zip(df["MMP_Label"], df["Yeo7_Network_Label"])),
            dict(zip(df["MMP_Label"], df["Yeo7_Network_ID"])))


def load_ner_subject_pairs():
    out = {}
    for sub in range(1, 37):
        for run in range(1, 25):
            p = f"{PART5}/sub-{sub:02d}/run-{run:02d}/run_summary.parquet"
            if not os.path.exists(p):
                continue
            z = np.load(f"{PART5}/sub-{sub:02d}/run-{run:02d}/onsets.npz")
            df = pd.read_parquet(p)
            for _, r in df.iterrows():
                key = (sub, r["seed_roi"], r["endpoint_roi"])
                d = out.setdefault(key, {"n_trials": 0, "n_sig": 0, "n_runs": 0, "onsets": []})
                d["n_trials"] += int(r["n_trials"])
                d["n_sig"] += int(r["n_significant"])
                d["n_runs"] += 1
                k = str(int(r["endpoint_channel_idx"]))
                if k in z and len(z[k]):
                    d["onsets"].extend(z[k].tolist())
    rows = []
    for (sub, a, b), d in out.items():
        rows.append({"subject": sub, "seed": a, "endpoint": b,
                     "n_trials": d["n_trials"], "n_runs": d["n_runs"],
                     "response_probability": d["n_sig"] / d["n_trials"],
                     "onset_ms_median": float(np.median(d["onsets"])) if d["onsets"] else np.nan})
    return pd.DataFrame(rows)


def build_units(prob, onset, ner, mats, regions, lab, iid):
    r2i = {r: i for i, r in enumerate(regions)}
    pairs = set()
    for _, r in ner.iterrows():
        si, oi = r2i[r["seed"]], r2i[r["endpoint"]]
        if np.isfinite(prob.iloc[si, oi]):
            pairs.add((r["seed"], r["endpoint"]))
    pairs = sorted(pairs)
    fam_of = lambda a, b: hashlib.sha1(f"{a}->{b}".encode()).hexdigest()[:12]
    fam16 = lambda a, b: hashlib.sha1(f"{a}->{b}".encode()).hexdigest()[:16]
    n_subj_support = {}
    for (a, b) in pairs:
        n_subj_support[(a, b)] = int(((ner["seed"] == a) & (ner["endpoint"] == b)).sum())
    units = []
    for (a, b) in pairs:
        si, oi = r2i[a], r2i[b]
        ner_rows = ner[(ner["seed"] == a) & (ner["endpoint"] == b)]
        tl = mats["tract_length"][si, oi]
        terc = 0 if tl < 104.86403219927423 else (1 if tl < 143.0989994869595 else 2)
        base = {"seed": a, "endpoint": b, "candidate_path_family": fam_of(a, b),
                "path_family_16": fam16(a, b), "path_length": 1,
                "seed_yeo7": lab.get(a.replace("_ROI", ""), "Unknown"),
                "end_yeo7": lab.get(b.replace("_ROI", ""), "Unknown"),
                "seed_yeo7_id": int(iid.get(a.replace("_ROI", ""), 8)),
                "end_yeo7_id": int(iid.get(b.replace("_ROI", ""), 8)),
                "seed_hemi": 1 if a.startswith("L") else 0,
                "end_hemi": 1 if b.startswith("L") else 0,
                "tract_length_mm": tl, "length_tercile": terc,
                "n_subjects_supporting_pair": n_subj_support[(a, b)],
                "nav_eff": mats["nav_eff"][si, oi], "rout_eff": mats["rout_eff"][si, oi],
                "search_info": mats["search_info"][si, oi],
                "comm_norm": mats["comm_norm"][si, oi]}
        u = dict(base)
        u.update({"domain": "F-TRACT", "subject": 0, "n_trials": np.nan, "n_runs": 1,
                  "response_probability": float(prob.iloc[si, oi]),
                  "onset_ms_median": float(onset.iloc[si, oi])
                  if np.isfinite(onset.iloc[si, oi]) else np.nan,
                  "onset_available": int(np.isfinite(onset.iloc[si, oi]))})
        units.append(u)
        for _, r in ner_rows.iterrows():
            u = dict(base)
            u.update({"domain": "NER", "subject": int(r["subject"]),
                      "n_trials": int(r["n_trials"]), "n_runs": int(r["n_runs"]),
                      "response_probability": float(r["response_probability"]),
                      "onset_ms_median": float(r["onset_ms_median"])
                      if np.isfinite(r["onset_ms_median"]) else np.nan,
                      "onset_available": int(np.isfinite(r["onset_ms_median"]))})
            units.append(u)
    U = pd.DataFrame(units)
    seeds = sorted(U["seed"].unique())
    U["fold"] = U["seed"].map({s: i for i, s in enumerate(seeds)})
    return U


def fit_models(U, C=1.0):
    """Leave-one-seed-out OOF fits for all MODEL_COLS; returns (model_rows, oof_all)."""
    y = (U["domain"] == "F-TRACT").astype(int).values
    ner_mask = (U["domain"] == "NER").values
    cnt_global = U[ner_mask].groupby(["seed", "endpoint"]).size().to_dict()
    cnt_by_fold = U[ner_mask].groupby(["seed", "endpoint", "fold"]).size().to_dict()
    oof = []
    model_rows = []
    for mname, cols in MODEL_COLS.items():
        X = U[cols].fillna(0.0).values
        probs = np.full(len(U), np.nan)
        for fold in U["fold"].unique():
            tr = (U["fold"] != fold).values
            te = (U["fold"] == fold).values
            w = np.ones(tr.sum())
            tr_idx = np.where(tr)[0]
            for pos, i in enumerate(tr_idx):
                if ner_mask[i]:
                    key = (U["seed"].iat[i], U["endpoint"].iat[i], fold)
                    n_ner = cnt_global[(U["seed"].iat[i], U["endpoint"].iat[i])] - cnt_by_fold.get(key, 0)
                    w[pos] = 1.0 / max(n_ner, 1)
            clf = LogisticRegression(max_iter=3000, C=C)
            clf.fit(X[tr], y[tr], sample_weight=w)
            probs[te] = clf.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y, probs)
        bacc = balanced_accuracy_score(y, (probs >= 0.5).astype(int))
        model_rows.append({"model": mname, "n_features": len(cols),
                           "oof_auc": auc, "oof_balanced_accuracy": bacc})
        oof.append(pd.DataFrame({"model": mname, "domain": U["domain"],
                                 "oof_prob": probs, "fold": U["fold"],
                                 "seed": U["seed"], "endpoint": U["endpoint"]}))
    oof_all = pd.concat(oof)
    return model_rows, oof_all


def main():
    os.makedirs(OUT, exist_ok=True)
    regions = load_region_order()
    mats = load_struct()
    prob, onset = load_ftract()
    lab, iid = load_yeo7()
    ner = load_ner_subject_pairs()
    U = build_units(prob, onset, ner, mats, regions, lab, iid)
    U.to_csv(f"{OUT}/COMMON_SUPPORT_UNITS.csv", index=False)

    bal = []
    for c in ["tract_length_mm", "n_subjects_supporting_pair", "nav_eff", "rout_eff",
              "search_info", "comm_norm"]:
        a = U[U.domain == "NER"][c].dropna()
        b = U[U.domain == "F-TRACT"][c].dropna()
        sd = np.sqrt((a.var() + b.var()) / 2)
        bal.append({"covariate": c, "ner_mean": a.mean(), "ftract_mean": b.mean(),
                    "std_pooled": sd, "smd": (a.mean() - b.mean()) / sd if sd > 0 else 0.0,
                    "n_ner_rows": len(a), "n_ftract_rows": len(b)})
    pd.DataFrame(bal).to_csv(f"{OUT}/MATCHING_BALANCE.csv", index=False)

    U[["seed", "endpoint", "candidate_path_family", "domain", "subject", "fold"]].to_csv(
        f"{OUT}/FOLD_ASSIGNMENTS.csv", index=False)

    model_rows, oof_all = fit_models(U)
    pd.concat([pd.DataFrame(model_rows)]).to_csv(f"{OUT}/_MODEL_ROWS.csv", index=False)
    oof_all.to_csv(f"{OUT}/OOF_PREDICTIONS.csv", index=False)
    y = (U["domain"] == "F-TRACT").astype(int).values
    seed_list = np.asarray(U["seed"])
    seeds = sorted(U["seed"].unique())
    auc_by_model = {r["model"]: r["oof_auc"] for r in model_rows}

    boot_aucs = {m: [] for m in MODEL_COLS}
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(seeds), len(seeds))
        keep = np.isin(seed_list, [seeds[i] for i in idx])
        if keep.sum() < 10 or (y[keep].sum() == 0) or ((1 - y[keep]).sum() == 0):
            continue
        for mname in MODEL_COLS:
            s = oof_all[oof_all.model == mname].iloc[np.where(keep)[0]]
            try:
                boot_aucs[mname].append(roc_auc_score(
                    (s.domain == "F-TRACT").astype(int), s.oof_prob))
            except ValueError:
                pass
    ci_rows = []
    for mname in MODEL_COLS:
        vals = np.asarray(boot_aucs[mname])
        ci_rows.append({"model": mname, "oof_auc": auc_by_model[mname],
                        "auc_ci_lo": np.percentile(vals, 2.5) if len(vals) else np.nan,
                        "auc_ci_hi": np.percentile(vals, 97.5) if len(vals) else np.nan})

    def probs_by(mname, keep):
        s = oof_all[oof_all.model == mname].iloc[np.where(keep)[0]]
        return s.oof_prob.values

    for base in ["C0", "C1"]:
        d = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, len(seeds), len(seeds))
            keep = np.isin(seed_list, [seeds[i] for i in idx])
            if keep.sum() < 10:
                continue
            a1 = roc_auc_score(y[keep], probs_by("C+P", keep))
            a0 = roc_auc_score(y[keep], probs_by(base, keep))
            d.append(a1 - a0)
        ci_rows.append({"model": f"C+P_minus_{base}",
                        "oof_auc": auc_by_model["C+P"] - auc_by_model[base],
                        "auc_ci_lo": np.percentile(d, 2.5) if d else np.nan,
                        "auc_ci_hi": np.percentile(d, 97.5) if d else np.nan})
    pd.DataFrame(ci_rows).to_csv(f"{OUT}/INCREMENTAL_MODEL_COMPARISON.csv", index=False)

    X = U[MODEL_COLS["C+P"]].fillna(0.0).values
    obs = auc_by_model["C+P"]
    cnt = 0
    for _ in range(N_PERM):
        yp = y.copy()
        for (a, b) in sorted(set(zip(U["seed"], U["endpoint"]))):
            m = (U["seed"] == a) & (U["endpoint"] == b)
            idx = np.where(m.values)[0]
            if len(idx) >= 2:
                yp[idx] = RNG.permutation(yp[idx])
        clf = LogisticRegression(max_iter=3000, C=1.0)
        try:
            clf.fit(X, yp)
            cnt += int(roc_auc_score(yp, clf.predict_proba(X)[:, 1]) >= obs)
        except ValueError:
            pass
    pd.DataFrame([{"model": "C+P", "observed_auc": obs, "n_permutations": N_PERM,
                   "n_ge_observed": cnt, "empirical_p": (cnt + 1) / (N_PERM + 1),
                   "procedure": "within-matched-pair domain-label swaps; refit per permutation; descriptive"}]
                 ).to_csv(f"{OUT}/PERMUTATION_RESULTS.csv", index=False)

    print(json.dumps({"n_units": len(U), "n_pairs": U.candidate_path_family.nunique(),
                      "n_seeds": len(seeds), "models": model_rows,
                      "delta_cp_vs_c0": auc_by_model["C+P"] - auc_by_model["C0"]}, indent=2))


if __name__ == "__main__":
    main()
