"""Independent phase-eliminated simulations of the two current selected fits.

No oscillator phase is borrowed from a full-model trajectory. The reduced
systems evolve their own D/V phases and memories, solving the instantaneous
stable X--SCN lag algebraically. Exact circular geometry and the coherent
approximation are evaluated separately without refitting.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

import architecture_model as original
import fit_protocol as fit
import parameter_registry
import reduced_model as reduced

ROOT = Path(__file__).resolve().parents[1]
STEP = 1/64
VARIANTS = ('oscillatory_X', 'adiabatic_exact_geometry', 'adiabatic_coherent')
OUTPUTS = ('reduction_parameters.json', 'reduction_targets.csv', 'reduction_traces.csv.gz')


def fingerprint():
    sha = hashlib.sha256()
    for name in ('unified_reduction.py', 'reduced_model.py', 'architecture_model.py'):
        sha.update((ROOT/'code'/name).read_bytes())
    for arch in ('D-period-X', 'coupling-X'):
        sha.update(parameter_registry.fingerprint(arch).encode())
    return sha.hexdigest()


def parameters():
    records = []
    for arch in ('D-period-X', 'coupling-X'):
        selected = parameter_registry.selected(arch)
        records.append(dict(architecture=arch, objective='main_protocol',
                            seed=selected['seed'], free_parameter_count=selected['free_parameter_count'],
                            vector=selected['vector'], fit_id=f"unified_{arch}_seed_{selected['seed']}",
                            selection_criterion=selected['selection_criterion'],
                            note='Current selected fit; independent reduction without refitting.'))
    return records


def reduced_root_residual(trace, p, mode, schedule, ex, coherent):
    """Evaluate the algebraic locking equation at recorded reduced states."""
    t, d, v, md, mx = (trace[:, j] for j in (0, 5, 6, 8, 10))
    delta = d-v; a = .8; b = .2
    resultant = np.sqrt(a*a+b*b+2*a*b*np.cos(delta))
    wd = a*(a+b*np.cos(delta))/(resultant*resultant); wv = 1-wd
    light = ((t < 0) & ((t+720) % schedule[0] < schedule[1])).astype(float)
    theta = np.where(t < 0, 2*np.pi*((t+720) % schedule[0])/schedule[0], 0.)
    td = np.clip(p[0]-p[3]*np.tanh(md), 16, 34) if mode == 0 else p[0]
    tx = np.clip(p[2]-p[4]*np.tanh(mx), 16, 34)
    kvd = p[9]+(p[17]*np.tanh(md) if mode == 1 else 0.)
    f0d = 2*np.pi/td+p[10]*np.sin(delta)+p[14]*light*np.sin(theta-d)
    f0v = 2*np.pi/p[1]+kvd*np.sin(delta)+p[15]*p[14]*light*np.sin(theta-v)
    factor = np.where((t >= 0) & ex, 0., 1.)
    if coherent:
        gg = factor*(wd*p[11]+wv*p[12])
        jj = factor*(-wd*p[11]*b+wv*p[12]*a)*np.sin(delta)
    else:
        gg = factor*(wd*p[11]*(a+b*np.cos(delta))+wv*p[12]*(b+a*np.cos(delta)))/resultant
        jj = factor*(-wd*p[11]*b+wv*p[12]*a)*np.sin(delta)/resultant
    alpha = trace[:, 24]
    residual = 2*np.pi/tx-wd*f0d-wv*f0v-(p[13]+gg)*np.sin(alpha)-jj*np.cos(alpha)
    stable_derivative = (p[13]+gg)*np.cos(alpha)-jj*np.sin(alpha)
    valid = trace[:, 25] > .5
    return dict(valid_root_residual_max=float(np.max(np.abs(residual[valid]))) if valid.any() else None,
                invalid_root_residual_max=float(np.max(np.abs(residual[~valid]))) if (~valid).any() else 0.,
                minimum_valid_contraction=float(np.min(stable_derivative[valid])) if valid.any() else None,
                sampled_invalid_roots=int((~valid).sum()))


def exact_identity_error(trace, p, ex):
    """Check the circular identity on full-model states, without approximation."""
    delta = trace[:, 5]-trace[:, 6]
    alpha = original.wrap(trace[:, 7]-trace[:, 20])
    r = trace[:, 19]
    factor = np.where((trace[:, 0] >= 0) & ex, 0., 1.)
    fd = factor*p[11]*(.8*np.sin(alpha)+.2*np.sin(alpha-delta))/r
    fv = factor*p[12]*(.8*np.sin(alpha+delta)+.2*np.sin(alpha))/r
    return max(float(np.max(np.abs(fd-trace[:, 16]))), float(np.max(np.abs(fv-trace[:, 17]))))


def generate(force=False):
    destination = ROOT/'data'
    checkpath = ROOT/'validation'/'reduction_validation.json'
    fp = fingerprint()
    if not force and checkpath.exists():
        prior = json.loads(checkpath.read_text(encoding='utf-8'))
        if prior.get('fingerprint') == fp and all((destination/name).exists() and hashlib.sha256((destination/name).read_bytes()).hexdigest() == prior.get('output_sha256', {}).get(name) for name in OUTPUTS):
            print('Using verified current-parameter reduction simulations.', flush=True)
            return prior
    selected = parameters()
    rows = []; trace_tables = []; score_rows = []; root_checks = []; half_checks = []; timescales = []
    max_identity = 0.
    for record in selected:
        p = np.asarray(record['vector']); arch = record['architecture']; mode = original.ARCHITECTURES[arch]
        for variant in VARIANTS:
            inv = np.empty((5, 10)); exmat = np.empty((5, 10))
            for i, condition in enumerate(original.ORDER):
                schedule = original.SCHEDULES[condition]
                for ex, matrix in ((False, inv), (True, exmat)):
                    context = 'exvivo' if ex else 'invivo'
                    if variant == 'oscillatory_X':
                        _, trace = original.simulate(arch, p, condition, ex=ex, dt=STEP)
                        invalid_ld = invalid_dd = 0
                        max_identity = max(max_identity, exact_identity_error(trace, p, ex))
                    else:
                        coherent = variant == 'adiabatic_coherent'
                        trace, invalid_ld, invalid_dd = reduced.run(p, mode, *schedule, STEP, ex, coherent)
                        root_check = reduced_root_residual(trace, p, mode, schedule, ex, coherent)
                        root_checks.append(dict(fit_id=record['fit_id'], variant=variant,
                                                condition=condition, context=context,
                                                invalid_LD_steps=int(invalid_ld), invalid_DD_steps=int(invalid_dd), **root_check))
                    metric, refined = reduced.metrics(trace); matrix[i] = metric
                    row = dict(fit_id=record['fit_id'], architecture=arch, objective=record['objective'],
                               variant=variant, condition=condition, context=context,
                               invalid_LD_steps=int(invalid_ld), invalid_DD_steps=int(invalid_dd))
                    row.update(zip(('behavior_h', 'D_h', 'V_h', 'X_h', 'gap_release_h', 'gap_7day_h'), map(float, metric[:6])))
                    row.update(zip(('behavior_refined_h', 'D_refined_h', 'V_refined_h'), map(float, refined)))
                    rows.append(row)
                    if variant == 'adiabatic_coherent':
                        half, bad_ld_half, bad_dd_half = reduced.run(p, mode, *schedule, STEP/2, ex, True)
                        half_metric, half_refined = reduced.metrics(half)
                        half_checks.append(dict(fit_id=record['fit_id'], condition=condition, context=context,
                                                maximum_period_step_difference_h=float(np.max(np.abs(np.asarray(refined)-half_refined))),
                                                gap_step_difference_h=float(abs(metric[5]-half_metric[5])),
                                                half_invalid_LD_steps=int(bad_ld_half), half_invalid_DD_steps=int(bad_dd_half)))
                        if not ex:
                            saved = pd.DataFrame(dict(time_from_DD_h=trace[:, 0], lag=trace[:, 24],
                                                      memory_X=trace[:, 10], contraction=trace[:, 26]))
                            for name in ('fit_id', 'variant', 'condition', 'context'):
                                saved[name] = row[name]
                            trace_tables.append(saved[['fit_id','variant','condition','context','time_from_DD_h','lag','memory_X','contraction']])
                            rates = trace[trace[:, 0] >= 0, 26]
                            valid_rates = rates[rates > 0]
                            timescales.append(dict(fit_id=record['fit_id'], architecture=arch, condition=condition,
                                                   minimum_relaxation_h=float(1/valid_rates.max()) if len(valid_rates) else None,
                                                   maximum_relaxation_h=float(1/valid_rates.min()) if len(valid_rates) else None,
                                                   invalid_DD_steps=int(invalid_dd)))
            score_rows.append(dict(fit_id=record['fit_id'], architecture=arch, variant=variant,
                                   shared_final_score=float(fit.final_score(inv, exmat)),
                                   note='Scores use saved 0.25-h output; the independently verified optimization score uses its specified fine-step phase summary.'))
            print(f"Reduction {arch}/{variant}: complete", flush=True)
    targets = pd.DataFrame(rows)
    comparison = targets.pivot(index=['fit_id','architecture','condition','context'], columns='variant', values=['behavior_refined_h','D_refined_h','V_refined_h','gap_7day_h'])
    discrepancies = []
    for variant in VARIANTS[1:]:
        for (fid, arch), group in comparison.groupby(level=['fit_id','architecture']):
            discrepancies.append(dict(fit_id=fid, architecture=arch, variant=variant,
                                      maximum_period_difference_from_full_h=max(float((group[(key,variant)]-group[(key,'oscillatory_X')]).abs().max()) for key in ('behavior_refined_h','D_refined_h','V_refined_h')),
                                      maximum_gap_difference_from_full_h=float((group[('gap_7day_h',variant)]-group[('gap_7day_h','oscillatory_X')]).abs().max())))
    (destination/'reduction_parameters.json').write_text(json.dumps(selected, indent=2)+'\n', encoding='utf-8')
    targets.to_csv(destination/'reduction_targets.csv', index=False)
    pd.concat(trace_tables, ignore_index=True).to_csv(destination/'reduction_traces.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    report = dict(fingerprint=fp, selected_fits=[dict(architecture=p['architecture'], seed=p['seed'], fit_id=p['fit_id']) for p in selected],
                  simulation_cases=len(rows), half_step_cases=len(half_checks), integration_step_h=STEP,
                  half_step_h=STEP/2, trace_sampling_h=.25,
                  maximum_period_step_difference_h=max(q['maximum_period_step_difference_h'] for q in half_checks),
                  maximum_gap_step_difference_h=max(q['gap_step_difference_h'] for q in half_checks),
                  root_residual_max=max(q['valid_root_residual_max'] or 0 for q in root_checks),
                  exact_identity_error_max=max_identity,
                  invalid_LD_steps=sum(q['invalid_LD_steps'] for q in root_checks),
                  invalid_DD_steps=sum(q['invalid_DD_steps'] for q in root_checks),
                  stable_root_checks=root_checks, half_step_checks=half_checks,
                  lag_relaxation_timescales=timescales, full_reduced_discrepancies=discrepancies,
                  shared_score_comparison=score_rows, new_optimization=False,
                  independent_X_phase_integrated=False, width_mm=175,
                  output_sha256={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in OUTPUTS})
    report['all_checks_passed'] = (report['invalid_LD_steps'] == report['invalid_DD_steps'] == 0
                                   and report['root_residual_max'] < 1e-12 and max_identity < 1e-12
                                   and report['maximum_period_step_difference_h'] < .01
                                   and report['maximum_gap_step_difference_h'] < .01)
    checkpath.parent.mkdir(exist_ok=True)
    checkpath.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('simulation_cases','half_step_cases','maximum_period_step_difference_h','maximum_gap_step_difference_h','invalid_LD_steps','invalid_DD_steps','all_checks_passed','full_reduced_discrepancies')}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--force', action='store_true')
    generate(force=ap.parse_args().force)
