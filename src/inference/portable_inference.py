from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["sample_id", "repeat", "outer_fold"]
PAIR_COLS = KEY + ["y_true", "y_pred"]
DRAW_TAG = "s10_review_linked"


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def _one_arm(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(frame, PAIR_COLS, label)
    arm = frame.loc[:, PAIR_COLS].copy()
    if (arm["sample_id"].astype(str).str.strip() == "").any():
        raise ValueError(f"{label}: empty sample_id")
    if arm[KEY + ["y_true", "y_pred"]].isna().any().any():
        raise ValueError(f"{label}: null key, outcome, or prediction")
    if arm[KEY].duplicated().any():
        raise ValueError(f"{label}: duplicate (sample_id, repeat, outer_fold) key")
    if arm[["sample_id", "repeat"]].duplicated().any():
        raise ValueError(f"{label}: more than one OOF prediction per sample/repeat")
    if not np.isfinite(arm[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: outcome or prediction is nonfinite")
    return arm.sort_values(["repeat", "outer_fold", "sample_id"], kind="stable").reset_index(drop=True)


def prepare(
    arm_a: pd.DataFrame, arm_b: pd.DataFrame, sample_map: pd.DataFrame
) -> pd.DataFrame:
    """Return paired OOF rows, rejecting unmatched/misaligned arms and repeats."""
    a = _one_arm(arm_a, "arm A")
    b = _one_arm(arm_b, "arm B")
    if not a[KEY].equals(b[KEY]):
        raise ValueError("OOF keys differ: incomplete or mismatched arm pairing")
    if not np.array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy()):
        raise ValueError("arm A and B outcomes differ for paired OOF keys")

    _require_columns(sample_map, ["sample_id", "seed_roi"], "sample map")
    mapping = sample_map.loc[:, ["sample_id", "seed_roi"]].copy()
    if (mapping["sample_id"].astype(str).str.strip() == "").any() or (mapping["seed_roi"].astype(str).str.strip() == "").any():
        raise ValueError("sample map has an empty sample_id or seed_roi")
    if mapping.isna().any().any() or mapping["sample_id"].duplicated().any():
        raise ValueError("sample map has missing or duplicate sample_id/seed_roi")
    if not set(a["sample_id"]).issubset(set(mapping["sample_id"])):
        raise ValueError("sample map misses OOF sample_id(s)")

    repeats = sorted(a["repeat"].unique())
    support = set(a["sample_id"])
    for repeat in repeats:
        subset = a.loc[a["repeat"] == repeat]
        if set(subset["sample_id"]) != support:
            raise ValueError("incomplete support: every repeat must contain every sample")
        if len(subset) < 3:
            raise ValueError("each repeat needs at least three OOF samples")
    if a.groupby("sample_id")["y_true"].nunique().max() != 1:
        raise ValueError("y_true changes for the same sample across repeats")

    paired = a.rename(columns={"y_pred": "pa"}).copy()
    paired["pb"] = b["y_pred"].to_numpy()
    paired = paired.merge(mapping, on="sample_id", how="left", validate="many_to_one", sort=False)
    if paired["seed_roi"].isna().any():
        raise ValueError("missing seed_roi after mapping")
    if paired.groupby(["repeat", "seed_roi"])["outer_fold"].nunique().max() != 1:
        raise ValueError("a seed_roi appears in multiple outer folds in one repeat")
    return paired


def wide_arms(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert C8R and C4 columns of a saved wide OOF table to aligned arms."""
    _require_columns(frame, KEY + ["y_true", "c4_pred", "c8r_pred"], "wide OOF")
    a = frame[KEY + ["y_true", "c8r_pred"]].rename(columns={"c8r_pred": "y_pred"})
    b = frame[KEY + ["y_true", "c4_pred"]].rename(columns={"c4_pred": "y_pred"})
    return a, b


def labeled_arms(frame: pd.DataFrame, label_a: str, label_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a saved arm-labeled OOF table into aligned process/endpoint rows."""
    _require_columns(frame, PAIR_COLS + ["arm"], "labeled OOF")
    if label_a == label_b or set(frame["arm"].dropna().unique()) != {label_a, label_b}:
        raise ValueError("labeled OOF must contain exactly the two selected arm labels")
    return (frame.loc[frame["arm"] == label_a, PAIR_COLS].copy(),
            frame.loc[frame["arm"] == label_b, PAIR_COLS].copy())


def _components(paired: pd.DataFrame, unit: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    if unit not in ("cluster", "sample"):
        raise ValueError("unit must be 'cluster' (primary) or 'sample' (sensitivity)")
    column = "seed_roi" if unit == "cluster" else "sample_id"
    # The first-appearance unit order matches perm_linked for a key-sorted input.
    codes, unit_names = pd.factorize(paired[column], sort=False)
    rep_codes, rep_names = pd.factorize(paired["repeat"], sort=False)
    y = paired["y_true"].to_numpy(dtype=float)
    a = paired["pa"].to_numpy(dtype=float)
    b = paired["pb"].to_numpy(dtype=float)
    sst = np.empty(len(rep_names), dtype=float)
    for i in range(len(rep_names)):
        vals = y[rep_codes == i]
        sst[i] = np.square(vals - vals.mean()).sum()
    if not np.isfinite(sst).all() or (sst <= 0).any():
        raise ValueError("repeat-pooled R2 undefined: nonpositive or nonfinite SST")
    a_err = np.square(y - a)
    b_err = np.square(y - b)
    n_units = len(unit_names)
    n_reps = len(rep_names)
    diff_by_repeat_unit = np.zeros((n_reps, n_units), dtype=float)
    np.add.at(diff_by_repeat_unit, (rep_codes, codes), b_err - a_err)
    contribution = np.sum(diff_by_repeat_unit / sst[:, None], axis=0) / n_reps
    r2_a = np.mean([1 - a_err[rep_codes == i].sum() / sst[i] for i in range(n_reps)])
    r2_b = np.mean([1 - b_err[rep_codes == i].sum() / sst[i] for i in range(n_reps)])
    return contribution, np.array([r2_a, r2_b]), sst, list(unit_names), list(rep_names)


def seed_from(*parts: object) -> int:
    """Original linked-test seed rule: SHA256(joined parts) first 8 bytes."""
    key = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def inference(paired: pd.DataFrame, unit: str, draws: int, tag: str) -> dict:
    """Two-sided 2*min(tails), with both historical and +1 MC versions."""
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("tag must be nonempty; use distinct tags for distinct contrasts")
    d, (r2_a, r2_b), _, names, repeats = _components(paired, unit)
    observed = float(r2_a - r2_b)
    if not np.isclose(d.sum(), observed, rtol=0, atol=1e-12):
        raise AssertionError("unit contributions do not match repeat-pooled R2 difference")
    lower = upper = 0
    for k in range(draws):
        rng = np.random.default_rng(seed_from(DRAW_TAG, unit, tag, k))
        signs = 1 - 2 * rng.integers(0, 2, size=len(names))
        null = float(signs @ d)
        lower += int(null <= observed)
        upper += int(null >= observed)
    return {
        "statistic": "mean over repeats of pooled R2(arm A) - pooled R2(arm B)",
        "unit": unit,
        "unit_column": "seed_roi" if unit == "cluster" else "sample_id",
        "exchange_across_all_repeats": True,
        "tag": tag,
        "draw_tag": DRAW_TAG,
        "draws": draws,
        "n_samples": int(paired["sample_id"].nunique()),
        "n_repeats": len(repeats),
        "n_rows": len(paired),
        "n_exchange_units": len(names),
        "r2_a": float(r2_a),
        "r2_b": float(r2_b),
        "delta_r2": observed,
        "lower_tail_count": lower,
        "upper_tail_count": upper,
        "p_two_sided": min(1.0, 2 * min(lower, upper) / draws),
        "p_two_sided_plus1": min(1.0, 2 * min(lower + 1, upper + 1) / (draws + 1)),
        "p_convention": "2*min(P(null<=obs),P(null>=obs)); +1 variant adds 1 to each tail count",
    }


def direct_exchange_delta(paired: pd.DataFrame, unit: str, flips: dict) -> float:
    """Slow row-level reference for synthetic tests; flips maps unit name to bool."""
    column = "seed_roi" if unit == "cluster" else "sample_id"
    missing = set(paired[column]) - set(flips)
    if missing:
        raise ValueError("flips missing one or more exchange units")
    mask = paired[column].map(flips).to_numpy(dtype=bool)
    a = np.where(mask, paired["pb"], paired["pa"])
    b = np.where(mask, paired["pa"], paired["pb"])
    vals = []
    for repeat in sorted(paired["repeat"].unique()):
        keep = (paired["repeat"] == repeat).to_numpy()
        y = paired.loc[keep, "y_true"].to_numpy(dtype=float)
        den = np.square(y - y.mean()).sum()
        vals.append((np.square(y - b[keep]).sum() - np.square(y - a[keep]).sum()) / den)
    return float(np.mean(vals))


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError("Parquet reading requires pyarrow or fastparquet") from exc


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Test paired held-out predictions with linked exchanges')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--oof", type=Path, help="wide Parquet with c4_pred and c8r_pred")
    inputs.add_argument("--oof-long", type=Path, help="single long Parquet with arm and y_pred")
    inputs.add_argument("--arm-a", type=Path, help="long C8R Parquet with y_pred")
    parser.add_argument("--arm-b", type=Path, help="long C4 Parquet with y_pred")
    parser.add_argument("--arm-a-label", default="C8R-All", help="process label in --oof-long")
    parser.add_argument("--arm-b-label", default="C4", help="endpoint label in --oof-long")
    parser.add_argument("--map", type=Path, help="CSV with sample_id and seed_roi")
    parser.add_argument("--unit", choices=["cluster", "sample"], default="cluster")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tag", help="nonempty contrast identifier; affects deterministic draws")
    parser.add_argument("--output", type=Path, help="write JSON; refuses overwrite")
    parser.add_argument("--self-test", action="store_true", help="run synthetic tests, no data I/O")
    args = parser.parse_args(argv)
    if args.self_test:
        from test_portable_inference import run_tests
        return run_tests()
    if not args.map or not args.output or not args.tag:
        parser.error("--map, --output and --tag are required for a data run")
    if bool(args.arm_a) != bool(args.arm_b) or not (args.oof or args.oof_long or args.arm_a):
        parser.error("supply --oof, --oof-long, OR both --arm-a and --arm-b")
    if (args.oof or args.oof_long) and args.arm_b:
        parser.error("--arm-b is only valid with --arm-a")
    if args.draws < 1:
        parser.error("--draws must be >= 1")
    if args.output.exists():
        parser.error(f"output exists; refusing overwrite: {args.output}")
    source_paths = [args.oof] if args.oof else ([args.oof_long] if args.oof_long else [args.arm_a, args.arm_b])
    source_paths.append(args.map)
    if any(not path.is_file() for path in source_paths):
        parser.error("all source paths must exist as files")
    if args.oof:
        a, b = wide_arms(_read_parquet(args.oof))
    elif args.oof_long:
        a, b = labeled_arms(_read_parquet(args.oof_long), args.arm_a_label, args.arm_b_label)
    else:
        a, b = _read_parquet(args.arm_a), _read_parquet(args.arm_b)
    paired = prepare(a, b, pd.read_csv(args.map))
    result = inference(paired, args.unit, args.draws, args.tag)
    result["source_sha256"] = {str(p): _sha256(p) for p in source_paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
