from common import *
from scipy.stats import pearsonr

FAMILIES = ['photoperiod', 'T-cycle', 'T-cycle x photoperiod']
HISTORY_KEYS = ['photoperiod_basis', 'developmental_photoperiod',
                'continuation_photoperiod', 'aggregation_basis']
PAIR_KEYS = ['study', 'year', 'age_label', 'condition_family', 'condition_label'] + HISTORY_KEYS


def matched_pairs(obs, scn):
    """Pair archived means within reporter as well as study, age and condition.

    Aton's behavior reporter is unspecified; its one selected SCN reporter is
    unambiguous. Never make a cross-reporter pair when both labels are known.
    """
    b = obs[obs.measurement.eq('Behavior')].copy()
    scn = scn.copy()
    for frame in [b, scn]:
        frame[HISTORY_KEYS] = frame[HISTORY_KEYS].fillna('')
    merged = b.merge(scn, on=PAIR_KEYS, suffixes=('_behavior', '_scn'))
    known = merged.reporter_behavior.notna()
    unknown_reporter_counts = merged[~known].groupby(PAIR_KEYS, dropna=False).reporter_scn.nunique()
    if (unknown_reporter_counts > 1).any():
        raise ValueError('Ambiguous SCN reporter for a behavioral observation with no reporter label')
    excluded = merged[known & merged.reporter_behavior.ne(merged.reporter_scn)]
    merged = merged[~known | merged.reporter_behavior.eq(merged.reporter_scn)].copy()
    q = merged[PAIR_KEYS].copy()
    for source, target in [('reporter_behavior', 'reporter_behavior'),
                           ('reporter_scn', 'reporter_scn'),
                           ('scn_region_scn', 'scn_region'),
                           ('age_category_behavior', 'age_category'),
                           ('photoperiod_scn', 'photoperiod'),
                           ('t_cycle_scn', 't_cycle'),
                           ('mean_period_h_behavior', 'behavior_mean_period_h'),
                           ('sem_h_behavior', 'behavior_sem_h'),
                           ('n_behavior', 'behavior_n'),
                           ('mean_period_h_scn', 'scn_mean_period_h'),
                           ('sem_h_scn', 'scn_sem_h'),
                           ('n_scn', 'scn_n'),
                           ('developmental_exposure_window_scn', 'developmental_exposure_window'),
                           ('continuation_duration_scn', 'continuation_duration'),
                           ('assay_age_scn', 'assay_age'),
                           ('exposure_start_age_scn', 'exposure_start_age'),
                           ('exposure_duration_scn', 'exposure_duration'),
                           ('entrainment_ld_cycles_behavior', 'behavior_entrainment_ld_cycles'),
                           ('entrainment_ld_cycles_scn', 'scn_entrainment_ld_cycles'),
                           ('measurement_window_behavior', 'behavior_measurement_window'),
                           ('measurement_window_scn', 'scn_measurement_window'),
                           ('source_sd_h_behavior', 'behavior_source_sd_h'),
                           ('source_sd_h_scn', 'scn_source_sd_h'),
                           ('uncertainty_basis_behavior', 'behavior_uncertainty_basis'),
                           ('uncertainty_basis_scn', 'scn_uncertainty_basis'),
                           ('source_reference_scn', 'source_reference')]:
        q[target] = merged[source]
    q = q.drop_duplicates().reset_index(drop=True)
    return q, excluded[PAIR_KEYS + ['reporter_behavior', 'reporter_scn']]


def correlation(z):
    if (len(z) >= 3 and z.behavior_mean_period_h.nunique() > 1
            and z.scn_mean_period_h.nunique() > 1):
        return float(pearsonr(z.behavior_mean_period_h, z.scn_mean_period_h).statistic)
    return np.nan


def main():
    obs = read('tcycle_period_observations.csv')
    scn = read('tcycle_selected_scn_observations.csv')
    archived_pairs = read('tcycle_selected_scn_behavior_correlation_pairs.csv')
    pairs, excluded = matched_pairs(obs, scn)
    studies = sorted(set(zip(obs.study, obs.year)))
    markers = {s: MARKERS[i] for i, s in enumerate(studies)}

    # Color keys sit inside the panels (age in B, reporter in D) as colored
    # labels, like the reporter labels of G to I. Study symbols and line
    # conventions are decoded in the manuscript caption. The footer strip of
    # the previous version is removed, so the figure is 14 mm shorter with
    # unchanged panel sizes.
    f = fig(170)
    axs = f.subplots(3, 3)
    f.subplots_adjust(left=.10, right=.985, top=.9306, bottom=.0800,
                      wspace=.30, hspace=.55)
    counts, correlations, line_records, clipping = [], [], [], []
    for j, family in enumerate(FAMILIES):
        b = obs[(obs.measurement == 'Behavior') & (obs.condition_family == family)]
        s = scn[scn.condition_family == family]
        conds = sorted(set(b.condition_label) | set(s.condition_label),
                       key=lambda x: ([float(n) for n in re.findall(r'\d+(?:\.\d+)?', x)],
                                      0 if x.startswith('SP ') else 1, x))
        for k, z in enumerate([b, s]):
            ax, labels = axs[k, j], []
            for i, c in enumerate(conds):
                g = z[z.condition_label == c]
                n = g[['study', 'year']].drop_duplicates().shape[0]
                label = c.replace(' LD', '').replace(' ', '\n')
                labels.append(label + '\n' + f'k={n}')
                counts.append(dict(panel=chr(65 + k * 3 + j), condition=c,
                                   independent_studies=n, means=len(g)))
                offsets = np.linspace(-.19, .19, len(g)) if len(g) > 1 else [0]
                for dx, (_, r) in zip(offsets, g.iterrows()):
                    color = AGE.get(r.age_category, '#666666') if k == 0 else REP.get(r.reporter, '#666666')
                    ax.errorbar(i + dx, r.mean_period_h, yerr=r.sem_h,
                                fmt=markers[(r.study, r.year)], color=color,
                                ms=4.5, alpha=.88, capsize=1.8, lw=1.0, mew=.85)
            ax.set(xticks=range(len(conds)), xticklabels=labels,
                   xlim=(-.55, len(conds) - .45),
                   ylim=(22, 25.3) if k == 0 else (18.5, 27))
            ax.tick_params(axis='x', labelsize=8, pad=3)
            ax.tick_params(axis='y', pad=2)
            if j == 0:
                ax.set_ylabel('Behavioral period (h)' if k == 0 else 'Ex vivo SCN period (h)')
            lo, hi = ax.get_ylim()
            errors = z.sem_h.fillna(0)
            clipping.append(dict(panel=chr(65 + k * 3 + j),
                                 n_points=len(z),
                                 clipped_errorbars=int(((z.mean_period_h-errors < lo)
                                                      | (z.mean_period_h+errors > hi)).sum())))
        ax = axs[2, j]
        g = pairs[pairs.condition_family == family].copy()
        for _, r in g.iterrows():
            ax.errorbar(r.behavior_mean_period_h, r.scn_mean_period_h,
                        xerr=r.behavior_sem_h, yerr=r.scn_sem_h,
                        fmt=markers[(r.study, r.year)], color=REP.get(r.reporter_scn, '#666666'),
                        ms=4.5, alpha=.9, capsize=1.8, lw=1.0, mew=.85, zorder=3)
        g['branch'] = g.photoperiod.fillna('') if family == FAMILIES[2] else ''
        # Preserve age strata; two-condition observations get a descriptive
        # connecting segment, not a statistically uninformative n=2 regression.
        for key, z in g.groupby(['study', 'year', 'reporter_scn', 'age_label', 'branch'], dropna=False):
            z = z.drop_duplicates(['condition_label', 'behavior_mean_period_h', 'scn_mean_period_h'])
            x = z.behavior_mean_period_h.to_numpy()
            y = z.scn_mean_period_h.to_numpy()
            record = dict(family=family, study=key[0], year=int(key[1]), reporter=key[2],
                          age=key[3], branch=key[4], n=len(z),
                          pearson_r=correlation(z), slope=np.nan, intercept=np.nan)
            if len(z) >= 3 and len(np.unique(x)) >= 3:
                sl, it = np.polyfit(x, y, 1)
                xs = np.array([x.min(), x.max()])
                style = '--' if key[-1] == 'LP' else '-'
                ax.plot(xs, sl*xs+it, color=REP.get(key[2], '#666666'),
                        alpha=.95, lw=1.4, ls=style, zorder=2)
                record.update(line_type='within-cohort regression', slope=sl, intercept=it)
            elif len(z) == 2 and len(np.unique(x)) == 2:
                order = np.argsort(x)
                ax.plot(x[order], y[order], color=REP.get(key[2], '#666666'),
                        alpha=.85, lw=1.25, ls=':', zorder=2)
                record['line_type'] = 'two-mean connector; no regression or correlation'
            else:
                record['line_type'] = 'no line: fewer than two distinct behavioral means'
            line_records.append(record)
        # Reporter summaries use the plotted means; line-group correlations are
        # separately supplied in Figure1_regression_groups.csv. I is split by
        # photoperiod so its two statistics correspond to its two fitted lines.
        stat_groups = g.groupby(['reporter_scn', 'branch']) if j == 2 else g.groupby('reporter_scn')
        for i, (key, z) in enumerate(stat_groups):
            reporter, branch = key if isinstance(key, tuple) else (key, '')
            n, rr = len(z), correlation(z)
            short = {'Bmal1-ELuc': 'Bmal1', 'PER2::LUC': 'PER2',
                     'Per1::GFP': 'Per1::GFP', 'Per1-GFP': 'Per1::GFP',
                     'Per1-luc': 'Per1-luc'}.get(reporter, reporter)
            label = f'{branch}: n={n}' if branch else f'{short}: n={n}'
            # r is not reported for two matched conditions (see legend)
            label += f', r={rr:.2f}' if np.isfinite(rr) else ''
            ax.text(.025, .98-.105*i, label, transform=ax.transAxes, va='top',
                    fontsize=8, color=REP.get(reporter, '#666666'),
                    bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=.4))
            correlations.append(dict(panel=chr(71+j), family=family, reporter=reporter,
                                     branch=branch, n_paired_condition_means=n,
                                     k_independent_studies=z[['study', 'year']].drop_duplicates().shape[0],
                                     pearson_r=rr,
                                     interpretation='Descriptive correlation of plotted condition means; clustered within studies/ages, no inferential p-value. n<3 suppressed.'))
        errors = g.scn_sem_h.fillna(0)
        lo = (g.scn_mean_period_h-errors).min()
        hi = (g.scn_mean_period_h+errors).max()
        span = hi-lo
        ax.set_ylim(lo-.10*span, hi+(.56 if j == 0 else .34)*span)
        ax.margins(x=.09)
        ax.set_xlabel('Behavioral period (h)', labelpad=3)
        if j == 0:
            ax.set_ylabel('Ex vivo SCN period (h)')
    for i, ax in enumerate(axs.flat):
        letter(ax, chr(65+i), x=-.16, y=1.025)
    for j, txt in enumerate(['Photoperiod', 'T-cycle', 'Photoperiod × T-cycle']):
        pos = axs[0, j].get_position()
        f.text((pos.x0+pos.x1)/2, .974, txt, ha='center', fontsize=9.2, weight='bold')
    # Color encodes age in A-C and reporter in D-I, not light schedule.
    # In-panel keys: colored labels without symbols, because symbol shape
    # already identifies the study. Age in B (all four age classes occur there,
    # upper left is empty); reporter in D (lower right is empty).
    box = dict(facecolor='white', alpha=.9, edgecolor='none', pad=.4)
    for i, (k, c) in enumerate(AGE.items()):
        axs[0, 1].text(.025, .98-.105*i, k.capitalize() if k != 'unknown' else 'Age unavailable',
                       transform=axs[0, 1].transAxes, va='top', ha='left', fontsize=8,
                       color=c, bbox=box, zorder=6)
    reporters = [(k, c) for k, c in REP.items() if k != 'Per1-GFP']
    for i, (k, c) in enumerate(reversed(reporters)):
        axs[1, 0].text(.975, .03+.105*i, k, transform=axs[1, 0].transAxes,
                       va='bottom', ha='right', fontsize=8, color=c, bbox=box, zorder=6)
    pd.DataFrame(counts).to_csv(TABLES/'Figure1_study_counts.csv', index=False)
    pd.DataFrame(correlations).to_csv(TABLES/'Figure1_reporter_correlations.csv', index=False)
    pd.DataFrame(line_records).to_csv(TABLES/'Figure1_regression_groups.csv', index=False)
    pairs.to_csv(TABLES/'Figure1_plotted_pairs.csv', index=False)
    # Selection flags refer to exactly the archived observations used in A-F.
    keys = ['study', 'year', 'age_label', 'reporter', 'condition_family', 'condition_label', 'scn_region']
    chosenkeys = set(tuple('' if pd.isna(x) else x for x in r) for r in scn[keys].itertuples(index=False, name=None))
    t = obs.copy()
    t.insert(0, 'observation_id', [f'OBS{i+1:03d}' for i in range(len(t))])
    t['species'] = np.where(t.strain.fillna('').str.contains('C57BL|CBA|M. musculus'), 'Mus musculus', 'not recorded')
    t['species_basis'] = np.where(t.strain.fillna('').str.contains('C57BL|CBA'), 'inferred from archived strain', np.where(t.strain.fillna('').eq('M. musculus'), 'explicit archived strain/species field', 'not available'))
    t['selected_for_Figure1'] = [r.measurement == 'Behavior' or tuple('' if pd.isna(x) else x for x in r[keys]) in chosenkeys for _, r in t.iterrows()]
    t['behavioral_n'] = np.where(t.measurement == 'Behavior', t.n, np.nan)
    t['SCN_n'] = np.where(t.measurement == 'SCN', t.n, np.nan)
    t['citation_year'] = t.year
    t.loc[t.study.str.startswith('Ciarleglio'), 'citation_year'] = 2011
    t['provenance_status'] = 'Archived extraction; not re-extracted from original articles in this package'
    confirmed = t.notes.fillna('').str.contains('Author-confirmed 2026-09-29', regex=False)
    t.loc[confirmed, 'provenance_status'] = 'Author-confirmed manuscript extraction revision, 2026-09-29; not independently re-extracted from original articles'
    reassigned = t.notes.fillna('').str.contains('Source/condition reconciled 2026-10-18', regex=False)
    t.loc[reassigned, 'provenance_status'] = 'Source/condition and sample size reconciled to original paper, 2026-10-18; manuscript mean/SEM retained, not re-digitized'
    photoperiod = t.notes.fillna('').str.contains('Photoperiod source reconciliation 2026-10-18', regex=False)
    t.loc[photoperiod, 'provenance_status'] = 'Photoperiod group identity and exposure history reconciled to original paper, 2026-10-18; numerical extraction retained, not re-digitized; see notes for uncertainty limitations'
    t.loc[t.study.str.startswith('Buijink'), 'provenance_status'] = 'Original supplemental Tables S2/S3 reconciled 2026-10-18: LP/SP labels, mean, SD and n verified; SEM derived as SD/sqrt(n)'
    source_checked = t.notes.fillna('').str.contains('Primary-source reconciliation 2026-11-25', regex=False)
    t.loc[source_checked & t.study.eq('Myung et al.'), 'provenance_status'] = 'Primary source reconciled 2026-11-25: Bmal1 behavioral n from Figure 4B; SP/LP regional source SD converted to SEM where applicable; means unchanged, see notes'
    t.loc[source_checked & t.study.str.startswith('Tackenberg'), 'provenance_status'] = 'Primary Methods and Figures 2/5 reconciled 2026-11-25: complete schedules, exposure/assay ages and readout-specific animal n; mean/SEM retained, not re-digitized'
    t.to_csv(TABLES/'Table_S1_literature_provenance.csv', index=False)
    audit = dict(
        pairing_keys=PAIR_KEYS + ['reporter when known'],
        unknown_behavior_reporter_policy='Allowed only for the single selected SCN reporter within the matching cohort; Aton2004.',
        archived_pair_count=len(archived_pairs), plotted_pair_count=len(pairs),
        plotted_pair_counts_by_family=pairs.groupby('condition_family').size().to_dict(),
        rejected_cross_reporter_joins=excluded.to_dict('records'),
        corrected_pair='Myung2012 Bmal1 behavior23.7h pairs with Bmal1 SCN23.9h, not PER2 SCN24.7h. Inputs preserved.',
        regression_count=sum(r['line_type']=='within-cohort regression' for r in line_records),
        two_mean_connector_count=sum(r['line_type'].startswith('two-mean') for r in line_records),
        omitted_single_mean_groups=json.loads(pd.DataFrame([r for r in line_records if r['line_type'].startswith('no line')]).to_json(orient='records')),
        categorical_errorbar_clipping=clipping,
        line_styles={'solid': 'within-cohort least-squares fit, n>=3; SP in I',
                     'dashed': 'LP within-cohort least-squares fit in I',
                     'dotted': 'two-mean descriptive connector; no fitted correlation'},
        correlation_policy='G/H summaries are descriptive reporter-level correlations of the plotted means; I reports SP/LP separately. Cohort-specific correlations and line eligibility are in the regression-groups table.'
    )
    (ROOT/'validation'/'figure1_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    save(f, 'Figure_1')


if __name__ == '__main__':
    main()
