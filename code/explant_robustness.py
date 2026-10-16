"""Direct T-cycle ordering under matched explant perturbations.

Uses full original-model endpoints already simulated at dt=1/64 h. Every
comparison pairs T22, LD and T26 under the same release intervention.
"""
from common import DATA,TABLES,ROOT,pd,np,json


def build():
    rows=[]
    for source,test,keys in [
        ('explant_coupling_sweep.csv','coupling',['coupling_scale']),
        ('explant_controls.csv','memory_phase',['coupling_scale','memory_D_retained','D_phase_advance_h'])]:
        data=pd.read_csv(DATA/source)
        for settings,g in data.groupby(keys):
            settings=settings if isinstance(settings,tuple) else (settings,)
            p=dict(zip(keys,settings));z=g.set_index('condition')
            for region in ['D','V']:
                periods={c:float(z.loc[c,region+'_h']) for c in ['T22','LD','T26']}
                rows.append(dict(test=test,region=region,
                                 coupling_percent=100*p['coupling_scale'],
                                 memory_retained_percent=100*p.get('memory_D_retained',1),
                                 D_phase_advance_h=p.get('D_phase_advance_h',0),
                                 T22_period_h=periods['T22'],LD_period_h=periods['LD'],T26_period_h=periods['T26'],
                                 T22_minus_LD_h=periods['T22']-periods['LD'],
                                 T26_minus_LD_h=periods['T26']-periods['LD'],
                                 T22_minus_T26_h=periods['T22']-periods['T26'],
                                 T22_gt_T26=periods['T22']>periods['T26'],
                                 T22_gt_LD_gt_T26=periods['T22']>periods['LD']>periods['T26']))
    out=pd.DataFrame(rows)
    out.to_csv(TABLES/'Figure3_explant_sensitivity.csv',index=False)
    ref=pd.read_csv(TABLES/'full_model_dynamic_endpoints.csv')
    intact=ref[~ref.ex_vivo].set_index('condition')
    baseline=float(intact.loc['T22','behavior_h']-intact.loc['T26','behavior_h'])
    summary=[]
    for (test,ret,region),g in out.groupby(['test','memory_retained_percent','region']):
        summary.append(dict(test=test,memory_retained_percent=ret,region=region,
                            combinations=len(g),T22_gt_T26_count=int(g.T22_gt_T26.sum()),
                            T22_gt_LD_gt_T26_count=int(g.T22_gt_LD_gt_T26.sum()),
                            contrast_min_h=float(g.T22_minus_T26_h.min()),
                            contrast_max_h=float(g.T22_minus_T26_h.max())))
    s=pd.DataFrame(summary);s.to_csv(TABLES/'Figure3_robustness_summary.csv',index=False)
    report=dict(source='Full original equations with selected penalized adaptive-D parameters; no refitting of perturbations',
                integration_step_h=.015625,period_record_h=336,
                intact_behavior_T22_minus_T26_h=baseline,
                coupling_scales=sorted(out.coupling_percent.unique().tolist()),
                memory_retained_percent=[0,50,100],D_phase_advance_h=[-2,0,2],
                comparison_rows=len(out),summary=summary,
                interpretation='Positive regional T22 minus T26 contrasts indicate the ex vivo ordering opposite to the negative intact behavioral contrast. Full memory reset leaves small positive residuals, not exact zero.')
    (ROOT/'validation/explant_robustness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return out,s,baseline




if __name__=='__main__':
    build()
    print('Updated direct explant-ordering tables and numerical checks.')
