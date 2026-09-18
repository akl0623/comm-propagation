#!/usr/bin/env python3
"""run_sensitivity.py - orchestration entry for the sensitivity analyses.

  1. SC density sensitivity (0.20 / 0.25 anchor / 0.30).
  2. Temporal binning sensitivity (2 ms / 5 ms bins).
  3. Propagation-complexity (variance) threshold sweep (T30-T70).
  4. Temporal-rule engine (RAW_STRICT full-precision) + V6 conditional CV.

See config/screening_params.json for the frozen thresholds.
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
    ('SC density 0.20/0.30 models',
     'p0_sensitivity_v1/sc_density/03_model/run_density_models.py'),
    ('Temporal bin width 2/5 ms',
     'p0_sensitivity_v1/bin_width/run_bin_width.py'),
    ('Variance-threshold sweep T30-T70',
     's10_optimized_v1/code/so_08_thresholds.py'),
    ('Temporal rule engine (RAW_STRICT)',
     'taskC_final_v1/code/taskC_phase08_temporal_build_v1.py'),
    ('V6 conditional sensitivity CV',
     'taskC_final_v1/code/taskC_phase08_v6_conditional_cv_v1.py'),
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
