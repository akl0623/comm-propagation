#!/usr/bin/env python3
"""run_insilico.py - orchestration entry for the in-silico feature-
displacement simulation.

  1. fixed_alpha_simulation.py: linear mixing df_sim = df*(1-alpha) +
     re_mean*alpha (FIXED_ALPHA = 0.242) toward per-target reference feature
     states.
  2. stage1_run_sr_analysis.py: SR = cosine similarity to the reference
     feature state, per target; alpha-sensitivity rows.
  3. stage2_run_negative_controls.py: negative controls + permutation test.

Library modules in this package:
  src/insilico/feature_displacement.py
  src/insilico/sr_analysis.py
  src/insilico/permutation_negative_controls.py
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
    ('Fixed-alpha linear-mixing simulation',
     'target_sim_alpha_correction_v1/02_fixed_alpha/code/fixed_alpha_simulation.py'),
    ('SR analysis (cosine similarity)',
     'target_sim_sr_focused_v1/code/stage1_run_sr_analysis.py'),
    ('Negative controls + permutation test',
     'target_sim_sr_focused_v1/code/stage2_run_negative_controls.py'),
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
