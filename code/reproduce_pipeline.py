"""One entry point for figures, all derived tables, and numerical checks.

Literature extractions and complete search finalist pools are bundled inputs.
No files from figures/ or tables/ are required to start. Model inputs are checked
against selected parameters and generator fingerprints. The default command repeats
all current scientific forward analyses, but does not repeat optimization.
"""
from pathlib import Path
import argparse,hashlib,json,time,platform

ROOT=Path(__file__).resolve().parents[1]
TABLE_NAMES=[
    'Figure1_plotted_pairs.csv','Figure1_regression_groups.csv',
    'Figure1_reporter_correlations.csv','Figure1_study_counts.csv',
    'Figure2_gain_sweep_convergence.csv','Figure2_selected_gain_operating_points.csv',
    'Figure3_explant_sensitivity.csv','Figure3_robustness_summary.csv',
    'Figure3K_absolute_period_summary.csv','fixed_period_Tcycle_contrasts.csv',
    'full_model_dynamic_endpoints.csv','Table_S1_literature_provenance.csv',
    'Table_S2_matched_searches.csv','optimization_reselection_audit.csv',
    'parameter_changes.csv','fit_target_errors.csv','fit_objective_components.csv',
    'Table_S3_fitted_parameters.csv','Table_S3_fixed_model_settings.csv',
    'memory_timescale_endpoints.csv','memory_timescale_summary.csv',
    'simple_coupling_endpoints.csv','simple_coupling_parameters.csv',
    'simple_coupling_shared_parameters.csv','simple_coupling_sensitivity.csv',
    'simple_coupling_perturbation_summary.csv','simple_coupling_grid.csv',
    'simple_coupling_convergence.csv','simple_coupling_search.csv',
    'simple_coupling_rule_parameters.csv','simple_coupling_rule_endpoints.csv',
    'simple_coupling_rule_diagnostics.csv',
]
FIGURE_PATHS=[f'figures/{section}/{name}.{ext}'
    for section,names in [('main',[f'Figure_{i}' for i in range(1,5)]),
                          ('supplement',[f'Figure_S{i}' for i in range(1,7)])]
    for name in names for ext in ['pdf','svg','png']]
FIGURE_PATHS+=['figures/supplement/Technical_appendix.pdf']
DOCUMENT_PATHS=[]
OUTPUT_PATHS=FIGURE_PATHS+['tables/'+n for n in TABLE_NAMES]+DOCUMENT_PATHS

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached',action='store_true',help='Reuse fingerprint-checked forward data; still rebuild every figure/table and run checks.')
    parser.add_argument('--verify',action='store_true',help='After rebuilding, independently rerun from only the 19 listed inputs in an isolated copy.')
    args=parser.parse_args(argv);args.refresh_model=not args.cached;started=time.perf_counter()
    manifest=json.loads((ROOT/'inputs.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path=ROOT/entry['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError('Required input missing or changed: '+entry['file'])
    for folder in ['tables','validation','figures/main','figures/supplement','work']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    missing_at_start=[s for s in OUTPUT_PATHS if not (ROOT/s).is_file()]
    print('1/4 Selecting penalized finalists and preparing all scientific data...',flush=True)
    import reselect_optimization
    reselect_optimization.select_all(apply=True)
    import parameter_tables
    parameter_tables.parameter_tables()
    # Import the model only after active selections have been refreshed.
    import full_model_data,architecture_sensitivity,unified_analyses,unified_reduction,memory_timescale,simple_coupling,simple_coupling_rule
    reintegrate=args.refresh_model
    full_model_data.main(force=args.refresh_model)
    unified_analyses.main(['--force'] if args.refresh_model else [])
    unified_reduction.generate(force=args.refresh_model)
    memory_timescale.generate(force=args.refresh_model)
    simple_coupling.generate(force=args.refresh_model)
    simple_coupling_rule.generate_rule(force=args.refresh_model)
    sweep=architecture_sensitivity.build(force=args.refresh_model)
    check_again=args.refresh_model or not (ROOT/'tables/Figure2_gain_sweep_convergence.csv').is_file()
    if check_again:
        print('Rebuilding Figure 2 step-halving table from full forward runs...',flush=True)
        architecture_sensitivity.validate_convergence(sweep)
    print('2/4 Rendering four main figures, S1-S6 and the appendix...',flush=True)
    from render_all import render_figures
    render_figures()
    print('3/4 Rebuilding remaining tables and checking numerical results...',flush=True)
    import validate_package
    validate_package.main()
    missing=[s for s in OUTPUT_PATHS if not (ROOT/s).is_file()]
    if missing:raise RuntimeError('Missing generated outputs: '+', '.join(missing))
    actual_tables={p.name for p in (ROOT/'tables').glob('*.csv')}
    if actual_tables!=set(TABLE_NAMES):raise RuntimeError('Unexpected table inventory: '+str(sorted(actual_tables.symmetric_difference(TABLE_NAMES))))
    from matplotlib import font_manager
    import pandas as pd
    font=Path(font_manager.findfont(font_manager.FontProperties(family=['Arial','DejaVu Sans'])))
    checks=json.loads((ROOT/'validation/package_checks.json').read_text())
    report=dict(passed=checks['passed'],scope='All publication figures and derived tables from bundled literature and complete search finalist pools; one selected adaptive-D parameter set for main/supporting analyses and separately selected comparison models. Forward-data provenance checked; no optimization rerun.',
        python=platform.python_version(),platform=platform.system(),font_family=font_manager.FontProperties(fname=str(font)).get_name(),
        font_sha256=hashlib.sha256(font.read_bytes()).hexdigest(),
        elapsed_seconds=round(time.perf_counter()-started,3),missing_outputs_at_start=missing_at_start,
        reference_histories_reintegrated=reintegrate,gain_convergence_recomputed=check_again,
        figures=len(FIGURE_PATHS),tables=len(TABLE_NAMES),documents=len(DOCUMENT_PATHS),numerical_checks=len(checks['checks']),
        outputs=[dict(file=s,bytes=(ROOT/s).stat().st_size,sha256=hashlib.sha256((ROOT/s).read_bytes()).hexdigest(),
                      **({'rows':len(pd.read_csv(ROOT/s))} if s.startswith('tables/') else {})) for s in OUTPUT_PATHS])
    (ROOT/'validation/reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"4/4 Complete: {len(FIGURE_PATHS)} figure files, {len(TABLE_NAMES)} tables, {len(checks['checks'])} checks passed.",flush=True)
    if args.verify:
        from verify_reproduction import verify
        verify()
    return report

if __name__=='__main__':main()
