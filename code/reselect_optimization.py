"""Select saved search finalists with the same observable score used in screening.

The searches, seeds, candidate sets and budgets are unchanged. Only final
selection is corrected: the archived fine-step Lomb--Scargle endpoints are
scored with the complete shared observable criterion, rather than the old
diagnostic compromise. No new parameter vectors are added to any candidate set.

Default execution writes an audit and prints the new winners without modifying
the active fit records. ``--apply`` writes the selected fit records. Source-pool
extraction is a one-time provenance operation; ordinary reproduction reads only
the compact immutable pools included in data/search_candidates.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd

import architecture_model as model
import fit_protocol as fit

ROOT = Path(__file__).resolve().parents[1]
POOL_ROOT = ROOT / 'data' / 'search_candidates'
OBJECTIVE_ID = 'shared_observable_penalty_v2'
JOBS = [(seed, arch) for seed in (1729, 2718, 3141)
        for arch in ('D-period-X', 'coupling-X', 'V-period-X', 'joint-DV-period-X')]
META_KEYS = ('architecture', 'objective', 'seed', 'free_parameter_count',
             'fingerprint', 'evaluations_screening', 'latin_samples',
             'local_starts', 'maximum_local_iterations', 'final_candidates',
             'elapsed_seconds', 'bounds')
ENDPOINT_COLUMNS = ('behavior_h', 'D_h', 'V_h', 'X_h',
                    'gap_release_folded_h', 'gap_7day_folded_arithmetic_h')


def vector_id(vector):
    return hashlib.sha256(json.dumps(vector, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def pool_path(seed, arch):
    return POOL_ROOT / f'seed_{seed}' / f'{arch}.json'


def preserve_former_selection(compact, prior):
    """A rerun must not redefine the before/after comparison's historical baseline."""
    for key in ('former_selected_vector', 'former_selected_diagnostic_score'):
        compact[key] = copy.deepcopy(prior[key])
    compact['former_selected_criterion'] = prior.get(
        'former_selected_criterion', prior.get('source_selection_criterion', 'legacy_final_diagnostic'))


def extract_pools(source_root, jobs=None, replace=False):
    """Preserve every finalist and its fine endpoints, omitting search histories."""
    source_root = Path(source_root).resolve()
    for seed, arch in JOBS if jobs is None else jobs:
        source = source_root / f'seed_{seed}' / f'fit_{arch}_main_protocol.json'
        raw = source.read_bytes()
        record = json.loads(raw)
        finalists = record['finalists']
        if not finalists:
            raise ValueError(f'Complete finalist pool missing: {source}')
        if len(finalists) != record['final_candidates']:
            raise ValueError(f'Candidate count differs from search record: {source}')
        compact = {key: record[key] for key in META_KEYS}
        compact.update(
            schema_version=1,
            source_relative_path=source.relative_to(source_root).as_posix(),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            search_status='completed search retained without modification',
            source_selection_criterion=record.get('selection_criterion','legacy_final_diagnostic'),
            coarse_step_h=fit.DT, fine_step_h=fit.DT_FINAL,
            former_selected_vector=record['vector'],
            former_selected_diagnostic_score=record['final_diagnostic_score'],
            former_selected_criterion=record.get('selection_criterion','legacy_final_diagnostic'),
            finalists=[])
        for candidate in finalists:
            c = copy.deepcopy(candidate)
            c['candidate_id'] = vector_id(c['vector'])
            # Keep this explicitly named historical value: it is no longer
            # the selection criterion, but remains auditable.
            c['historical_diagnostic_score'] = c.pop('final_diagnostic_score')
            compact['finalists'].append(c)
        target = pool_path(seed, arch)
        if target.exists() and replace:
            preserve_former_selection(compact, json.loads(target.read_text(encoding='utf-8')))
        text = json.dumps(compact, indent=2, allow_nan=False) + '\n'
        if target.exists() and target.read_text(encoding='utf-8') != text and not replace:
            raise RuntimeError(f'Refusing to replace an immutable candidate pool: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')


def endpoint_arrays(candidate):
    """Rebuild the shared scoring inputs from explicitly recorded observables."""
    inv = np.full((len(model.ORDER), 10), np.nan)
    ex = np.full_like(inv, np.nan)
    seen = set()
    for row in candidate['endpoints']:
        context = bool(row['ex_vivo'])
        key = (row['condition'], context)
        if key in seen:
            raise ValueError(f'Duplicate endpoint row: {key}')
        seen.add(key)
        array = ex if context else inv
        i = model.ORDER.index(row['condition'])
        array[i, :6] = [row[column] for column in ENDPOINT_COLUMNS]
    expected = {(condition, context) for condition in model.ORDER
                for context in (False, True)}
    if seen != expected or not np.isfinite(inv[:, :6]).all() or not np.isfinite(ex[:, :6]).all():
        raise ValueError('Each finalist must contain all finite in vivo/ex vivo endpoints')
    return inv, ex


def select_pool(path):
    raw = path.read_bytes()
    pool = json.loads(raw)
    scores = []
    maximum_score_difference = 0.0
    for candidate in pool['finalists']:
        if candidate['candidate_id'] != vector_id(candidate['vector']):
            raise ValueError(f'Candidate vector identity changed: {path}')
        inv, ex = endpoint_arrays(candidate)
        value = float(fit.screening_score(inv, ex, False))
        recorded = candidate['fine_screening_score']
        delta = abs(value - recorded)
        maximum_score_difference = max(maximum_score_difference, delta)
        if not math.isclose(value, recorded, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f'Reconstructed shared score differs: {path}: {value} vs {recorded}')
        scores.append(value)
    # Python's stable minimum preserves the archived candidate order at ties.
    index = min(range(len(scores)), key=scores.__getitem__)
    best = copy.deepcopy(pool['finalists'][index])
    source_criterion = pool.get('source_selection_criterion','legacy_final_diagnostic')
    best['source_recorded_final_score'] = best['historical_diagnostic_score']
    best['source_recorded_score_criterion'] = source_criterion
    # Older compact pools use this field name for whatever their source called
    # final_diagnostic_score. Compute the actual legacy diagnostic explicitly,
    # so new localization pools cannot masquerade as old diagnostic scores.
    inv, ex = endpoint_arrays(best)
    best['historical_diagnostic_score'] = float(fit.legacy_final_score(inv, ex))
    best['historical_diagnostic_score_criterion'] = 'legacy_final_diagnostic'
    best['final_diagnostic_score'] = scores[index]
    best['selection_objective'] = OBJECTIVE_ID
    result = {key: pool[key] for key in META_KEYS}
    result.update(
        historical_search_fingerprint=pool['fingerprint'],
        fingerprint=hashlib.sha256((pool['fingerprint'] + OBJECTIVE_ID).encode()).hexdigest(),
        selection_objective=OBJECTIVE_ID,
        selection_criterion=OBJECTIVE_ID,
        search_status=('same completed search; corrected final selection' if pool.get('source_selection_criterion','legacy_final_diagnostic')=='legacy_final_diagnostic' else 'completed search using shared observable penalties'),
        source_selection_criterion=pool.get('source_selection_criterion','legacy_final_diagnostic'),
        selected=best, vector=best['vector'],
        parameters={name: best['vector'][model.NAMES.index(name)]
                    for name in model.FREE[pool['architecture']]},
        final_diagnostic_score=scores[index],
        selected_origin=best['origin'],
        selected_candidate_id=best['candidate_id'],
        candidate_pool_path=path.relative_to(ROOT).as_posix(),
        candidate_pool_sha256=hashlib.sha256(raw).hexdigest(),
        all_candidate_scores_verified=True,
        maximum_score_reconstruction_error=maximum_score_difference,
        former_selected_vector=pool['former_selected_vector'],
        former_selected_diagnostic_score=pool['former_selected_diagnostic_score'],
        former_selected_criterion=pool.get('former_selected_criterion', source_criterion),
        selected_parameters_changed=(best['vector'] != pool['former_selected_vector']),
        finalists_retained_in=result_pool_ref(path))
    return result


def result_pool_ref(path):
    return path.relative_to(ROOT).as_posix()


def select_all(write_audit=True, apply=False):
    selected = [select_pool(pool_path(seed, arch)) for seed, arch in JOBS]
    rows = []
    for result in selected:
        former = np.asarray(result['former_selected_vector'])
        current = np.asarray(result['vector'])
        rows.append(dict(
            architecture=result['architecture'], seed=result['seed'],
            shared_final_score=result['final_diagnostic_score'],
            legacy_diagnostic_of_new_winner=result['selected']['historical_diagnostic_score'],
            former_selected_score=result['former_selected_diagnostic_score'],
            former_selected_score_criterion=result['former_selected_criterion'],
            source_pool_score_criterion=result['source_selection_criterion'],
            candidate_count=result['final_candidates'],
            selected_origin=result['selected_origin'],
            selected_parameters_changed=result['selected_parameters_changed'],
            changed_parameter_count=int(np.count_nonzero(current != former)),
            maximum_absolute_parameter_change=float(np.max(np.abs(current-former))),
            maximum_score_reconstruction_error=result['maximum_score_reconstruction_error'],
            pool_sha256=result['candidate_pool_sha256']))
        if apply:
            target = ROOT / 'data' / 'optimization' / f"seed_{result['seed']}" / f"fit_{result['architecture']}_main_protocol.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if write_audit:
        (ROOT/'tables').mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(ROOT/'tables'/'optimization_reselection_audit.csv', index=False)
        write_parameter_changes(selected)
        (ROOT/'validation').mkdir(exist_ok=True)
        audit = dict(objective=OBJECTIVE_ID, search_repeated=False,
                     explanation='This selection step scores all bundled fine-step finalists with the complete shared screening criterion; it adds no vectors or evaluations. D/coupling and seed-1729 localization pools came from previous searches; V/joint seeds 2718 and 3141 were added for three-seed localization coverage.',
                     pools_checked=len(rows), candidates_checked=sum(r['candidate_count'] for r in rows),
                     maximum_score_reconstruction_error=max(r['maximum_score_reconstruction_error'] for r in rows),
                     active_records_written=bool(apply), results=rows)
        (ROOT/'validation'/'optimization_reselection.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    return selected


def write_parameter_changes(selected):
    """Report parameter changes without conflating seeds, fits or objectives."""
    winners = {arch: min((q for q in selected if q['architecture'] == arch),
                        key=lambda q: q['final_diagnostic_score'])
               for arch in ('D-period-X', 'coupling-X')}
    former = {arch: min((q for q in selected if q['architecture'] == arch),
                       key=lambda q: q['former_selected_diagnostic_score'])
              for arch in winners}
    cases = [
        ('original_to_unified_D', 'D-period-X', fit.archive_vector(), 'original reference', winners['D-period-X']),
        ('previous_refitted_D_to_unified_D', 'D-period-X', former['D-period-X']['former_selected_vector'], str(former['D-period-X']['seed']), winners['D-period-X']),
        ('previous_refitted_K_to_unified_K', 'coupling-X', former['coupling-X']['former_selected_vector'], str(former['coupling-X']['seed']), winners['coupling-X'])]
    rows = []
    for comparison, architecture, previous, old_seed, current in cases:
        for name in model.FREE[architecture]:
            j = model.NAMES.index(name)
            old, new = float(previous[j]), float(current['vector'][j])
            rows.append(dict(comparison=comparison, architecture=architecture,
                             old_seed=old_seed, new_seed=current['seed'],
                             parameter=name, previous_value=old, selected_value=new,
                             delta=new-old, absolute_delta=abs(new-old), changed=(new != old)))
    pd.DataFrame(rows).to_csv(ROOT/'tables'/'parameter_changes.csv', index=False)


def validate_selected_dynamics(selected):
    """Independently integrate every selected vector with the final estimator."""
    rows = []
    for q in selected:
        inv, ex, endpoints = fit.rescore(np.asarray(q['vector']), q['architecture'])
        fresh = float(fit.final_score(inv, ex, False))
        recorded_inv, recorded_ex = endpoint_arrays(q['selected'])
        largest_endpoint_delta = max(float(np.max(np.abs(inv[:, :6]-recorded_inv[:, :6]))),
                                     float(np.max(np.abs(ex[:, :6]-recorded_ex[:, :6]))))
        if not math.isclose(fresh, q['final_diagnostic_score'], rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"Fresh selected score differs for {q['architecture']}/{q['seed']}")
        if largest_endpoint_delta > 1e-9:
            raise ValueError(f"Fresh selected endpoints differ for {q['architecture']}/{q['seed']}")
        rows.append(dict(architecture=q['architecture'], seed=q['seed'],
                         archived_score=q['final_diagnostic_score'], fresh_score=fresh,
                         maximum_endpoint_difference=largest_endpoint_delta, passed=True))
    (ROOT/'validation'/'optimization_selected_dynamics.json').write_text(
        json.dumps(dict(step_h=fit.DT_FINAL, criterion=OBJECTIVE_ID,
                        selected_records_checked=len(rows), results=rows), indent=2)+'\n', encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-source', type=Path, help='One-time source optimization directory; omitted in ordinary reproduction')
    parser.add_argument('--apply', action='store_true', help='Write corrected selections to active optimization records')
    parser.add_argument('--verify-dynamics', action='store_true', help='Independently simulate every selected vector with the fine-step estimator')
    args = parser.parse_args()
    if args.extract_source:
        extract_pools(args.extract_source)
    results = select_all(apply=args.apply)
    if args.verify_dynamics:
        validate_selected_dynamics(results)
    for q in results:
        print(f"{q['architecture']:20s} seed {q['seed']}: {q['final_diagnostic_score']:.10g}; "
              f"{q['final_candidates']} finalists; parameters changed={q['selected_parameters_changed']}")


if __name__ == '__main__':
    main()
