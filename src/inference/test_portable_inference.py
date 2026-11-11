import hashlib
import unittest

import numpy as np
import pandas as pd

from portable_inference import (
    DRAW_TAG,
    _components,
    direct_exchange_delta,
    inference,
    labeled_arms,
    prepare,
    seed_from,
    wide_arms,
)


def synthetic():
    samples = [("a0", "A", 0.0), ("a1", "A", 0.3), ("b0", "B", 0.7), ("b1", "B", 1.0)]
    rows_a, rows_b = [], []
    for rep in (0, 1):
        for sample, seed, y in samples:
            fold = 0 if seed == "A" else 1
            pa = y + (0.04 if rep == 0 else -0.03)
            pb = y + (0.20 if seed == "A" else -0.14)
            key = {"sample_id": sample, "repeat": rep, "outer_fold": fold, "y_true": y}
            rows_a.append({**key, "y_pred": pa})
            rows_b.append({**key, "y_pred": pb})
    mapping = pd.DataFrame([{"sample_id": sid, "seed_roi": seed} for sid, seed, _ in samples])
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b), mapping


class PortableInferenceTest(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.map = synthetic()

    def test_unit_contributions_equal_direct_row_exchanges(self):
        paired = prepare(self.a, self.b, self.map)
        for unit in ("cluster", "sample"):
            d, (r2_a, r2_b), _, names, repeats = _components(paired, unit)
            self.assertEqual(repeats, [0, 1])
            self.assertAlmostEqual(float(d.sum()), float(r2_a - r2_b), places=12)
            for mask in range(1 << len(names)):
                flips = {name: bool(mask & (1 << i)) for i, name in enumerate(names)}
                signs = np.array([-1 if flips[name] else 1 for name in names])
                self.assertAlmostEqual(float(signs @ d), direct_exchange_delta(paired, unit, flips), places=12)

    def test_two_sided_tail_counts_and_fixed_seed(self):
        paired = prepare(self.a, self.b, self.map)
        for unit in ("cluster", "sample"):
            first = inference(paired, unit, 37, "synthetic-C8R-C4")
            self.assertEqual(first, inference(paired, unit, 37, "synthetic-C8R-C4"))
            d, _, _, names, _ = _components(paired, unit)
            obs = first["delta_r2"]
            null = []
            for k in range(37):
                g = np.random.default_rng(seed_from(DRAW_TAG, unit, "synthetic-C8R-C4", k))
                bits = g.integers(0, 2, size=len(names))
                null.append(d @ (1 - 2 * bits))
            lower = sum(v <= obs for v in null)
            upper = sum(v >= obs for v in null)
            self.assertEqual((first["lower_tail_count"], first["upper_tail_count"]), (lower, upper))
            self.assertEqual(first["p_two_sided"], min(1, 2 * min(lower, upper) / 37))
            self.assertEqual(first["p_two_sided_plus1"], min(1, 2 * min(lower + 1, upper + 1) / 38))
            self.assertGreaterEqual(first["p_two_sided_plus1"], 2 / 38)
        self.assertEqual(seed_from(DRAW_TAG, "cluster", "demo", 0), int.from_bytes(hashlib.sha256(b"s10_review_linked|cluster|demo|0").digest()[:8], "big"))

    def test_ties_and_p_boundary(self):
        same = prepare(self.a, self.a, self.map)
        result = inference(same, "cluster", 5, "equal")
        self.assertEqual(result["delta_r2"], 0)
        self.assertEqual(result["lower_tail_count"], 5)
        self.assertEqual(result["upper_tail_count"], 5)
        self.assertEqual(result["p_two_sided"], 1)
        self.assertEqual(result["p_two_sided_plus1"], 1)

    def test_wide_format(self):
        wide = self.a.rename(columns={"y_pred": "c8r_pred"}).copy()
        wide["c4_pred"] = self.b["y_pred"].to_numpy()
        a, b = wide_arms(wide)
        paired = prepare(a, b, self.map)
        self.assertEqual(len(paired), 8)
        self.assertEqual(paired.seed_roi.nunique(), 2)

    def test_labeled_fullcv_oof(self):
        long_oof = pd.concat([self.a.assign(arm="C8R-All"), self.b.assign(arm="C4")], ignore_index=True)
        a, b = labeled_arms(long_oof, "C8R-All", "C4")
        paired = prepare(a, b, self.map)
        self.assertEqual(len(paired), 8)
        self.assertAlmostEqual(inference(paired, "cluster", 10, "tag")["delta_r2"],
                               inference(prepare(self.a, self.b, self.map), "cluster", 10, "tag")["delta_r2"])
        with self.assertRaisesRegex(ValueError, "exactly the two"):
            labeled_arms(pd.concat([long_oof, long_oof.iloc[[0]].assign(arm="extra")]), "C8R-All", "C4")

    def test_reject_duplicate_and_incomplete_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare(pd.concat([self.a, self.a.iloc[[0]]]), self.b, self.map)
        with self.assertRaisesRegex(ValueError, "differ"):
            prepare(self.a, self.b.iloc[1:], self.map)
        bad = self.a.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError, "incomplete support"):
            prepare(bad, bad, self.map)
        mismatch = self.b.copy()
        mismatch.loc[0, "outer_fold"] = 1
        with self.assertRaisesRegex(ValueError, "differ"):
            prepare(self.a, mismatch, self.map)
        bad_y = self.b.copy()
        bad_y.loc[0, "y_true"] = .9
        with self.assertRaisesRegex(ValueError, "outcomes differ"):
            prepare(self.a, bad_y, self.map)

    def test_reject_incorrect_map_support_or_group_folds(self):
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            prepare(self.a, self.b, pd.concat([self.map, self.map.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "misses OOF"):
            prepare(self.a, self.b, self.map.iloc[1:])
        bad_fold = self.a.copy()
        bad_fold.loc[bad_fold.sample_id == "a0", "outer_fold"] = 1
        with self.assertRaisesRegex(ValueError, "multiple outer folds"):
            prepare(bad_fold, bad_fold, self.map)


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PortableInferenceTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_tests())
