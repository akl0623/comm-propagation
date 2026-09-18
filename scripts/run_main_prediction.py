#!/usr/bin/env python3
"""run_main_prediction.py - orchestration entry for the main prediction
experiment (F-TRACT Task C, C4 vs C6 vs C8R nested grouped CV).

Invokes the canonical analysis scripts under PIPELINE_ROOT. To run, the
frozen inputs must be in place:
  - fold manifest: taskC0R2D_universe_freeze/taskC_primary_fold_manifest_C0R2D_v2.csv
  - features/targets as built by the feature-builder scripts
  - frozen env: python 3.12.2, numpy 2.5.1, pandas 3.0.0, scikit-learn 1.8.0
    (single-thread BLAS; see environment.yml)

Library modules in this package:
  src/models/elasticnet.py
  src/features/endpoint_features.py / legendre_features.py
  src/utils/cross_validation.py   (fold allocation, repeat_r2)
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
    ('Phase03 features (C4/C6)', 'taskC_final_v1/code/taskC_feature_builder_v1.py'),
    ('Phase03 features (C8R ordered basis)', 'taskC_final_v1/code/taskC_c8r_feature_builder_v3.py'),
    ('Phase04 nested grouped CV definition', 'taskC_final_v1/code/taskC_nested_cv_v1.py'),
    ('Phase05 production (500 cells, KKT gate)', 'taskC_final_v1/code/taskC_phase05_production_v4.py'),
    ('Phase05 assembly (OOF long table)', 'taskC_final_v1/code/taskC_phase05_assembly_v3.py'),
    ('Phase09 statistics (bootstrap CI, Holm)', 'taskC_final_v1/code/taskC_phase09_statistics_run_v1.py'),
]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true',
                    help='print the steps without executing')
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
