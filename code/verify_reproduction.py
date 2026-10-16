"""Rebuild in isolation from immutable inputs and compare numerical artifacts."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def compare_file(a, b):
    """CSV/JSON tolerate roundoff; figures and other artifacts compare bytes."""
    if not b.is_file():
        return dict(passed=False, comparison='missing rebuilt file')
    if sha(a) == sha(b):
        return dict(passed=True, comparison='identical bytes')
    try:
        if a.name.endswith(('.csv', '.csv.gz')):
            pd.testing.assert_frame_equal(pd.read_csv(a), pd.read_csv(b),
                                          check_exact=False, rtol=1e-10, atol=1e-10)
            return dict(passed=True, comparison='numerically equal CSV', tolerance=1e-10)
        if a.suffix == '.json':
            passed = json_equal(json.loads(a.read_text(encoding='utf-8')),
                                json.loads(b.read_text(encoding='utf-8')))
            return dict(passed=passed, comparison='JSON values', tolerance=1e-10)
    except (AssertionError, ValueError, TypeError) as exc:
        return dict(passed=False, comparison='numerical mismatch', detail=str(exc)[:1500])
    return dict(passed=False, comparison='different bytes')


def compare_outputs(reference, rebuilt):
    from reproduce_pipeline import OUTPUT_PATHS
    paths = list(OUTPUT_PATHS)
    paths += [p.relative_to(reference).as_posix()
              for p in sorted((reference/'data').rglob('*')) if p.is_file()]
    records = [dict(file=name, **compare_file(reference/name, rebuilt/name)) for name in paths]
    expected_data = {p.relative_to(reference/'data').as_posix()
                     for p in (reference/'data').rglob('*') if p.is_file()}
    actual_data = {p.relative_to(rebuilt/'data').as_posix()
                   for p in (rebuilt/'data').rglob('*') if p.is_file()}
    difference = sorted(expected_data.symmetric_difference(actual_data))
    checks_path = rebuilt/'validation/package_checks.json'
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {'passed': False, 'checks': []}
    return dict(passed=all(x['passed'] for x in records) and not difference and checks['passed'],
                figures=sum(x['file'].startswith('figures/') for x in records),
                tables=sum(x['file'].startswith('tables/') for x in records),
                data_files=sum(x['file'].startswith('data/') for x in records),
                numerical_checks=len(checks['checks']), data_inventory_difference=difference,
                files=records)


def verify():
    """Run without any generated data, figures, tables, reports or old caches."""
    from reproduce_pipeline import OUTPUT_PATHS
    missing = [name for name in OUTPUT_PATHS if not (ROOT/name).is_file()]
    if missing:
        raise RuntimeError('Build the reference outputs first: '+', '.join(missing))
    inputs = json.loads((ROOT/'inputs.json').read_text(encoding='utf-8'))['files']
    work = (ROOT/'work').resolve()
    work.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='clean-reproduction-', dir=work)).resolve()
    clean = temporary/'package'
    clean.mkdir()
    report = dict(passed=False, input_files=len(inputs), copied_generated_files=0,
                  fresh_forward_integrations=True, independent_process=True)
    start = time.perf_counter()
    try:
        shutil.copytree(ROOT/'code', clean/'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('reproduce.py', 'inputs.json', 'requirements.txt', 'requirements-lock.txt'):
            shutil.copy2(ROOT/name, clean/name)
        for entry in inputs:
            target = (clean/entry['file']).resolve()
            if not target.is_relative_to(clean.resolve()):
                raise ValueError('Input path escapes package: '+entry['file'])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/entry['file'], target)
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment['NUMBA_CACHE_DIR'] = str(temporary/'numba-cache')
        environment['MPLCONFIGDIR'] = str(temporary/'matplotlib-cache')
        print('Verification: rebuilding independently from immutable inputs; this is a second full run.', flush=True)
        process = subprocess.run([sys.executable, '-I', '-u', str(clean/'reproduce.py')],
                                 cwd=temporary, env=environment, check=False)
        report['exit_code'] = process.returncode
        if process.returncode == 0:
            report.update(compare_outputs(ROOT, clean))
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-start, 3)
        (ROOT/'validation/portability.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        # Only the newly created, resolved temporary directory is eligible.
        if temporary.is_relative_to(work) and temporary != work and temporary.name.startswith('clean-reproduction-'):
            shutil.rmtree(temporary)
    if not report['passed']:
        raise AssertionError('Clean reproduction differed; see validation/portability.json. Figure byte comparison also requires identical fonts.')
    print(f"Clean reproduction passed: {report['figures']} figure files, {report['tables']} tables, {report['data_files']} data files.", flush=True)
    return report
