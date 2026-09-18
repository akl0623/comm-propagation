#!/usr/bin/env python3
"""run_domain_classification.py - orchestration entry for Task B domain
classification (F-TRACT vs NER), V1-V4 design.

  tb01 features -> tb02 V1 mixed-CV baseline -> tb03 V2 LODO -> tb04 V3
  matched -> tb05 V4 confounder decomposition -> tb06 summary tables.

V4 (tb05_v4_confound.py) implements the confounder decomposition with
M1 confounder-only / M2 communication / M3 combined / M4 endpoint feature
sets; classifier = LogisticRegression(C=1.0) primary + RandomForestClassifier
(500 trees) sensitivity. NOTE: SVM is not used (documented design decision;
see README). Library module in this package:
  src/models/domain_classifier.py
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
BASE = os.path.join(PIPELINE_ROOT, 'taskB_classification_redesign_v2', 'code')

STEPS = [
    ('tb01: 94-dim features', 'tb01_build_features.py'),
    ('tb02: V1 mixed 5x5 CV baseline', 'tb02_v1_mixed_cv.py'),
    ('tb03: V2 leave-one-dataset-out', 'tb03_v2_lodo.py'),
    ('tb04: V3 matched sampling CV', 'tb04_v3_matched.py'),
    ('tb05: V4 confounder decomposition (LR + RF)', 'tb05_v4_confound.py'),
    ('tb06: summary tables', 'tb06_summary_tables.py'),
]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--from-step', type=int, default=0)
    args = ap.parse_args(argv)
    for i, (label, rel) in enumerate(STEPS):
        path = os.path.join(BASE, rel)
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
