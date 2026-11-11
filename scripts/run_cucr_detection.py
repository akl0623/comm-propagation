import argparse
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'src' / 'cucr_preprocessing' / 'ccep_detection_legacy.py'


def main():
    parser = argparse.ArgumentParser(description='Run cohort-level 0-100 ms CUCR response detection on authorized data')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--execute', action='store_true', help='explicitly run patient-data batch detection')
    options = parser.parse_args()
    if not SCRIPT.is_file():
        parser.error(f'missing source: {SCRIPT}')
    if not options.execute:
        print('Bundled 0-100 ms batch source verified; no data accessed. --execute requires patient-data access.')
        return 0
    root = options.data_root
    if not root or not root.is_dir():
        parser.error('--execute requires --data-root pointing to the authorized CUCR staging directory')
    if not (root / 'subject_electrodes.csv').is_file():
        parser.error('missing subject_electrodes.csv')
    files = list((root / 'output').glob('sub-*_task-ccepcoreg_run-*_parcel_seeg_data.npy'))
    if len(files) != 323:
        parser.error(f'expected 323 prepared subject/run arrays; found {len(files)}')
    import re
    missing = []
    for path in files:
        match = re.fullmatch(r'sub-(\d+)_task-ccepcoreg_run-(\d+)_parcel_seeg_data\.npy', path.name)
        if not match:
            missing.append(path.name)
            continue
        subject, run = match.groups()
        active = root / 'output' / f'sub-{subject}_task-ccepcoreg_run-{run}_active_parcels.csv'
        mapping = root / 'output' / f'sub-{subject}_task-ccepcoreg_run-{run}_electrode_to_parcel_mapping.csv'
        fallback = root / 'output' / f'sub-{subject}_task-ccepcoreg_run-01_electrode_to_parcel_mapping.csv'
        if not active.is_file() or not (mapping.is_file() or fallback.is_file()):
            missing.append(path.name)
    if missing:
        parser.error(f'{len(missing)} prepared runs lack active-parcel or mapping inputs')
    log_path = root / 'cucr_batch_100ms_release.log'
    if log_path.exists():
        parser.error(f'existing log would be overwritten: {log_path}')
    with log_path.open('x', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(SCRIPT)], cwd=str(root.resolve()),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    text = log_path.read_text(encoding='utf-8')
    completed = re.search(r'成功处理了\s*(\d+)\s*个文件', text)
    if not completed or int(completed.group(1)) != 323:
        parser.error(f'batch processed fewer than 323 runs; inspect authorized log: {log_path}')
    import numpy as np
    expected = {'response_probability_matrix_251114.txt': 7332,
                'median_onset_time_matrix_251114.txt': 6635}
    for name, count in expected.items():
        output = root / name
        if not output.is_file():
            parser.error(f'expected output missing: {name}')
        values = np.loadtxt(output)
        if values.shape != (360, 360) or int(np.isfinite(values).sum()) != count:
            parser.error(f'{name}: expected 360x360 matrix with {count} finite cells')
    print('Verified 323 processed runs and both cohort-level matrix support counts; log:', log_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
