#!/usr/bin/env python3
"""run_controls.py - orchestration entry for the negative-control analyses.

Invokes the canonical control scripts under PIPELINE_ROOT:

  1. Screen-level controls (stage4b_controls.py): order_shuffle,
     latency_permutation (LatencyNull), sc_rewiring, matched_random.
  2. C10 SC-null (taskC_phase07_sc_null_v1.py): degree-preserving rewiring +
     recomputation of the four communication matrices.
  3. Path-level controls (pl_04_full.py): SCNull / LatencyNull /
     OrderShuffle / EdgeShuffle / MatchedRandom / S10Path variants used in
     Fig.3C.

Library modules in this package:
  src/controls/screen_controls.py  (order_shuffle / matched_random / latency_null)
  src/controls/sc_null.py          (C10 SC-null)
"""
import argparse
import os
import subprocess
import sys

PIPELINE_ROOT = os.environ.get('PIPELINE_ROOT', '')
if not PIPELINE_ROOT:
    raise EnvironmentError(
        "PIPELINE_ROOT environment variable must be set to the data workspace root. "
        "See README.md (Data Availability section) for data download instructions."
    )

STEPS = [
    ('Screen-level negative controls (4 types)',
     'screen_match_confirm_v1/confirmatory/code/stage4b_controls.py'),
    ('C10 SC-null control',
     'taskC_final_v1/code/taskC_phase07_sc_null_v1.py'),
    ('Path-level controls (S10Path experiment)',
     's10_pathlevel_v1/code/pl_04_full.py'),
]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--from-step', type=int, default=0)
    args = ap.parse_args(argv)
    for i, (label, rel) in enumerate(STEPS):
        path = os.path.join(PIPELINE_ROOT, rel)
        if i < args.from_step:
            print(f'[skip {i}] {label}: {rel}')
            continue
        print(f'[step {i}] {label}: {rel}', flush=True)
        if args.dry_run:
            continue
        if not os.path.exists(path):
            sys.exit(f'ERROR: script not found at {path} '
                     f'(set PIPELINE_ROOT to the workspace root)')
        subprocess.run([sys.executable, path],
                       cwd=os.path.dirname(path), check=False)


if __name__ == '__main__':
    main(sys.argv[1:])
