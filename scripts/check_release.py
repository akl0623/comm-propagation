import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from release_runner import WORKFLOWS


def main():
    files = [p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts]
    failures = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            if ast.get_docstring(tree) is not None:
                failures.append(f'unclean module header: {path.relative_to(ROOT)}')
            lines = path.read_text(encoding='utf-8').splitlines()
            i = 1 if lines and lines[0].startswith('#!') else 0
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].lstrip().startswith('#'):
                failures.append(f'leading comment block: {path.relative_to(ROOT)}')
        except (SyntaxError, UnicodeDecodeError) as exc:
            failures.append(f'{path.relative_to(ROOT)}: {exc}')
    for steps in WORKFLOWS.values():
        for _, rel in steps:
            if not (ROOT / 'pipeline' / rel).is_file():
                failures.append(f'missing step: {rel}')
    if (ROOT / 'pipeline/taskC_final_v1').exists() or (ROOT / 'src/code').exists():
        failures.append('older conditional-pilot or duplicate compatibility tree remains')
    science = json.loads((ROOT / 'config/final_analysis.json').read_text(encoding='utf-8'))
    for key in ['primary_prediction', 'threshold_sensitivity', 'revised_inference', 'simulation', 'cohort_preprocessing', 'anatomical_enrichment', 'controls', 'domain_classification']:
        path = science[key]['entry']
        if not (ROOT / path).is_file():failures.append('missing final config entry: '+path)
    cucr = (ROOT/'src/cucr_preprocessing/ccep_detection_legacy.py').read_text(encoding='utf-8')
    if 'response_mask = (time >= 0) & (time <= 100)' not in cucr:
        failures.append('100-ms CUCR response window not located')
    if 'recompute_permutation_review' in (ROOT/'pipeline/perm_review_v1/code/run_threshold_fullcv.py').read_text(encoding='utf-8'):
        failures.append('old audit permutation dependency remains')
    result_ext = {'.parquet', '.csv', '.tsv', '.npy', '.npz', '.pkl', '.pickle',
                  '.pyc', '.log', '.out', '.err', '.txt', '.xlsx', '.png',
                  '.pdf', '.zip', '.h5', '.hdf5', '.mat'}
    extras=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
            and p.suffix.lower() in result_ext]
    if extras:failures.extend('data or cache included: '+x for x in extras)
    print(f"{'PASS' if not failures else 'FAIL'}: {len(files)} Python files, "
          f"{sum(map(len, WORKFLOWS.values()))} bundled workflow stages; "
          f"patient data in source: {len(extras)}")
    for issue in failures:print(' -',issue)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
