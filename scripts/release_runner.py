from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PKG = Path(__file__).resolve().parents[1]
PIPELINE = PKG / 'pipeline'
CODE_DIRS = sorted(p for p in PIPELINE.rglob('code') if p.is_dir())
WORKFLOWS = {
    'main_prediction': [
        ('S10 support from frozen attrition', 'iterative_combined_screen_v1/full_experiment/code/ps_support.py'),
        ('S10 C4/C6/C8R grouped 5x5 OOF', 'iterative_combined_screen_v1/full_experiment/code/ps_taskC.py'),
        ('S10 seed-cluster bootstrap and contrasts', 'iterative_combined_screen_v1/full_experiment/code/ps_boot.py'),
    ],
    'sensitivity': [
        ('SC density sensitivity', 'p0_sensitivity_v1/sc_density/03_model/run_density_models.py'),
        ('Temporal bin width sensitivity', 'p0_sensitivity_v1/bin_width/run_bin_width.py'),
        ('Full five-repeat threshold sensitivity', 'perm_review_v1/code/run_threshold_fullcv.py'),
    ],
    'controls': [
        ('S10 screen-level controls', 'iterative_combined_screen_v1/full_experiment/code/ps_controls.py'),
        ('S10 path-level controls', 's10_pathlevel_v1/code/pl_04_full.py'),
    ],
    'domain_classification': [
        ('Task B feature preparation', 'taskB_classification_redesign_v2/code/tb01_build_features.py'),
        ('V1 mixed CV baseline', 'taskB_classification_redesign_v2/code/tb02_v1_mixed_cv.py'),
        ('V2 leave-one-dataset-out', 'taskB_classification_redesign_v2/code/tb03_v2_lodo.py'),
        ('V3 matched sampling', 'taskB_classification_redesign_v2/code/tb04_v3_matched.py'),
        ('V4 confounder decomposition', 'taskB_classification_redesign_v2/code/tb05_v4_confound.py'),
        ('Task B summary', 'taskB_classification_redesign_v2/code/tb06_summary_tables.py'),
    ],
    'insilico': [
        ('Fixed-coefficient feature displacement', 'target_sim_alpha_correction_v1/02_fixed_alpha/code/fixed_alpha_simulation.py'),
        ('Build SR full-path cache and frozen target list', 'target_sim_sr_focused_v1/code/stage1_build_cache.py'),
        ('SR cosine analysis', 'target_sim_sr_focused_v1/code/stage1_run_sr_analysis.py'),
        ('SR negative controls', 'target_sim_sr_focused_v1/code/stage2_run_negative_controls.py'),
    ],
    'anatomical_enrichment': [
        ('S10 four-circuit enrichment', 'iterative_combined_screen_v1/full_experiment/code/ps_validation.py'),
        ('B4 four-circuit enrichment', 'b4_confirmatory_v1/code/b4_validation.py'),
        ('B5 four-circuit enrichment', 'screen_match_confirm_v1/confirmatory/code/stage4b_validation.py'),
    ],
}


def run(name: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Inspect or explicitly execute ' + name)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', help='verify bundled code paths without reading data')
    group.add_argument('--dry-run', action='store_true', help='show bundled steps; no execution')
    group.add_argument('--execute', action='store_true', help='run this data-dependent workflow explicitly')
    parser.add_argument('--from-step', type=int, default=0)
    parser.add_argument('--step', type=int, help='execute one indexed stage only')
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('PIPELINE_ROOT', '')) if os.environ.get('PIPELINE_ROOT') else None)
    options = parser.parse_args(argv)
    steps = WORKFLOWS[name]
    if options.from_step < 0 or options.from_step >= len(steps):
        parser.error(f'--from-step must be in [0,{len(steps) - 1}]')
    if options.step is not None and (options.step < 0 or options.step >= len(steps) or options.from_step != 0):
        parser.error('--step must name one valid index and cannot be combined with --from-step')
    missing = []
    for i, (label, rel) in enumerate(steps):
        script = PIPELINE / rel
        print(f'[{i}] {label}: {script.relative_to(PKG)}' + (' [missing]' if not script.is_file() else ''), flush=True)
        if not script.is_file():
            missing.append(rel)
    if missing:
        parser.error('required bundled scripts missing: ' + ', '.join(missing))
    if not options.execute:
        print('Code paths verified; no models or data were accessed. Execution requires --execute.', flush=True)
        return 0
    if options.data_root is None or not options.data_root.is_dir():
        parser.error('--execute requires --data-root /authorized/frozen/workspace (or PIPELINE_ROOT)')
    root = options.data_root.resolve()
    env = os.environ.copy()
    env['PIPELINE_ROOT'] = str(root)
    if name in ('main_prediction', 'controls'):
        env['PS_SCREEN'] = 'S10'
        print('Frozen screen: S10', flush=True)
    env['PYTHONPATH'] = os.pathsep.join([str(p) for p in CODE_DIRS] + [env.get('PYTHONPATH', '')])
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    selected = [steps[options.step]] if options.step is not None else steps[options.from_step:]
    for _, rel in selected:
        script = PIPELINE / rel
        print('Executing:', script.relative_to(PKG), flush=True)
        # Production scripts may write to the authorized data workspace.
        subprocess.run([sys.executable, str(script)], cwd=str(script.parent), env=env, check=True)
    return 0
