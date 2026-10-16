"""Fixed-fit sensitivity to the assumed light-history memory relaxation times.

The selected adaptive-D parameters are held fixed. Both memory constants are
changed throughout entrainment and DD; separate D-only and X-only sweeps isolate
their effects. This is a sensitivity test, not a fit or identification of 240 h.
"""
from pathlib import Path
import argparse, ast, hashlib, json
import numpy as np
import pandas as pd
import reference_model as model
import parameter_registry
from fit_protocol import final_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; TABLES=ROOT/'tables'; VALIDATION=ROOT/'validation'
TIMES=[60.,120.,180.,240.,360.,480.,720.]
MODES=['both','D_only','X_only']
DT=1/64
FILES=['memory_timescale_endpoints.csv','memory_timescale_scores.csv','memory_timescale_step_halving.csv']
TABLE_NAMES=['memory_timescale_endpoints.csv','memory_timescale_summary.csv']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint():
    """Hash numerical work independently of figure-only presentation changes."""
    h=hashlib.sha256(parameter_registry.fingerprint().encode())
    h.update(json.dumps({k:globals()[k] for k in ['TIMES','MODES','DT','FILES','TABLE_NAMES']},sort_keys=True).encode())
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    for name in ['endpoint','generate']:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        h.update(ast.dump(node,include_attributes=False).encode())
    for name in ['reference_model.py','fit_protocol.py']:
        h.update((ROOT/'code'/name).read_bytes())
    return h.hexdigest()


def endpoint(a):
    out=model.metrics(a)
    dd=a[a[:,0]>=0]; first=dd[dd[:,0]<=168+1e-8]
    light=a[(a[:,0]<0)&(a[:,0]>=-168)]
    # Exact final-selection grid is kept distinct from interpolated plotting
    # estimates. The six score inputs below reproduce fit_protocol.rescore.
    score=np.zeros(10)
    for j,col in enumerate([4,1,2,3]):
        score[j]=model.period(a,col,refined=False)
    score[4]=dd[0,13]; score[5]=first[:,13].mean()
    score[6]=first[:,10].mean(); score[7]=first[:,11].mean()
    score[8]=light[:,8].mean(); score[9]=light[:,9].mean()
    out.update(behavior_discrete_h=score[0],D_discrete_h=score[1],
               V_discrete_h=score[2],X_discrete_h=score[3],
               gap_release_folded_h=score[4],gap_7day_folded_arithmetic_h=score[5],
               memory_D_release=float(dd[0,8]),memory_X_release=float(dd[0,9]),
               memory_D_14day=float(dd[-1,8]),memory_X_14day=float(dd[-1,9]))
    return out,score


def export_tables():
    TABLES.mkdir(exist_ok=True)
    pd.read_csv(DATA/FILES[0]).to_csv(TABLES/TABLE_NAMES[0],index=False)
    pd.read_csv(DATA/FILES[1]).to_csv(TABLES/TABLE_NAMES[1],index=False)


def generate(force=False):
    report_path=VALIDATION/'memory_timescale.json'; fp=fingerprint()
    if not force and report_path.exists():
        old=json.loads(report_path.read_text(encoding='utf-8'))
        if old.get('fingerprint')==fp and all((DATA/n).exists() and old.get('data_sha256',{}).get(n)==sha(DATA/n) for n in FILES):
            export_tables(); return old
    targets=pd.read_csv(DATA/'archived_targets.csv').set_index('condition')
    rows=[]; scores=[]; checkrows=[]; cache={}
    for mode in MODES:
        for tau in TIMES:
            td=tau if mode in ('both','D_only') else 240.
            tx=tau if mode in ('both','X_only') else 240.
            inv=np.empty((5,10)); exv=np.empty((5,10)); case=[]
            for i,c in enumerate(model.ORDER):
                for ex,arr in [(False,inv),(True,exv)]:
                    key=(td,tx,c,ex)
                    if key not in cache:
                        a=model.simulate(c,ex=ex,dt=DT,changes={'mem_tau_d':td,'mem_tau_x':tx})
                        z,s=endpoint(a); cache[key]=(z,s)
                        if mode=='both' and tau in [60.,240.,720.]:
                            b=model.simulate(c,ex=ex,dt=DT/2,changes={'mem_tau_d':td,'mem_tau_x':tx})
                            fine=model.metrics(b)
                            pe=max(abs(z[k]-fine[k]) for k in ['behavior_h','D_h','V_h'])
                            ge=abs((z['gap_days1_7_circular_h']-fine['gap_days1_7_circular_h']+12)%24-12)
                            checkrows.append(dict(memory_tau_h=tau,condition=c,ex_vivo=ex,dt_h=DT,half_dt_h=DT/2,maximum_period_difference_h=pe,circular_gap_difference_h=ge,passed=pe<.005 and ge<.05))
                    z,s=cache[key]; arr[i]=s
                    row=dict(mode=mode,varied_tau_h=tau,memory_tau_D_h=td,memory_tau_X_h=tx,condition=c,ex_vivo=ex,**z)
                    rows.append(row);case.append(row)
            q=pd.DataFrame(case); ii=q[~q.ex_vivo].set_index('condition');ee=q[q.ex_vivo].set_index('condition')
            scored=dict(mode=mode,varied_tau_h=tau,memory_tau_D_h=td,memory_tau_X_h=tx,
                common_penalized_objective=float(final_score(inv,exv)),
                behavior_target_RMSE_h=float(np.sqrt(np.mean((ii.behavior_h-targets.behavior_mean_h)**2))),
                exvivo_D_target_RMSE_h=float(np.sqrt(np.mean((ee.D_h-targets.exvivo_scn_target_h)**2))),
                exvivo_V_target_RMSE_h=float(np.sqrt(np.mean((ee.V_h-targets.exvivo_scn_target_h)**2))),
                circular_phase_target_RMSE_h=float(np.sqrt(np.mean(((ee.gap_days1_7_circular_h-targets.gap_mean_h+12)%24-12)**2))),
                behavior_T22_minus_T26_h=float(ii.loc['T22','behavior_h']-ii.loc['T26','behavior_h']),
                exvivo_D_T22_minus_T26_h=float(ee.loc['T22','D_h']-ee.loc['T26','D_h']),
                exvivo_V_T22_minus_T26_h=float(ee.loc['T22','V_h']-ee.loc['T26','V_h']),
                exvivo_D_order_T22_LD_T26=bool(ee.loc['T22','D_h']>ee.loc['LD','D_h']>ee.loc['T26','D_h']),
                exvivo_V_order_T22_LD_T26=bool(ee.loc['T22','V_h']>ee.loc['LD','V_h']>ee.loc['T26','V_h']))
            scored['inversion_both_regions']=bool(scored['behavior_T22_minus_T26_h']<0 and scored['exvivo_D_T22_minus_T26_h']>0 and scored['exvivo_V_T22_minus_T26_h']>0)
            scores.append(scored)
            print(f'Memory sensitivity {mode}, {tau:g} h: score {scored["common_penalized_objective"]:.6g}',flush=True)
    endpoints=pd.DataFrame(rows); summary=pd.DataFrame(scores)
    for name,table in zip(FILES,[endpoints,summary,pd.DataFrame(checkrows)]):table.to_csv(DATA/name,index=False)
    baseline=summary[summary['mode'].eq('both')&summary.varied_tau_h.eq(240)].iloc[0]
    selected=parameter_registry.selected(); difference=abs(float(baseline.common_penalized_objective)-selected['final_diagnostic_score'])
    best=summary.loc[summary['mode'].eq('both')].sort_values('common_penalized_objective').iloc[0]
    report=dict(fingerprint=fp,parameter_fingerprint=parameter_registry.fingerprint(),parameter_source='best penalized D-period-X',selected_seed=selected['seed'],
        fixed_parameters=model.PARAMS,memory_time_constants_h=TIMES,modes=MODES,unique_full_simulations=len(cache),endpoint_rows=len(rows),score_rows=len(scores),
        exposure_h=720,DD_h=336,integration_step_h=DT,half_step_h=DT/2,
        scoring='Unchanged common penalized objective on original discrete Lomb-Scargle grid; plotted endpoints use refined Lomb-Scargle peaks and circular first-7-day phase gaps.',
        interpretation='Fixed-parameter sensitivity only. Other parameters are not reoptimized at each time constant; this test does not identify 240 h or establish it as an optimal biological constant.',
        baseline_objective=float(baseline.common_penalized_objective),selected_objective=selected['final_diagnostic_score'],baseline_objective_absolute_difference=difference,
        best_tested_shared_tau_h=float(best.varied_tau_h),best_tested_shared_objective=float(best.common_penalized_objective),
        common_tau_inversion=[dict(tau_h=float(r['varied_tau_h']),inversion=bool(r['inversion_both_regions']),D_order=bool(r['exvivo_D_order_T22_LD_T26']),V_order=bool(r['exvivo_V_order_T22_LD_T26'])) for r in scores if r['mode']=='both'],
        step_halving_cases=len(checkrows),maximum_period_difference_h=max(r['maximum_period_difference_h'] for r in checkrows),maximum_circular_gap_difference_h=max(r['circular_gap_difference_h'] for r in checkrows),
        data_sha256={n:sha(DATA/n) for n in FILES},passed=bool(difference<1e-5 and all(r['passed'] for r in checkrows)))
    VALIDATION.mkdir(exist_ok=True);report_path.write_text(json.dumps(report,indent=2),encoding='utf-8'); export_tables()
    if not report['passed']:raise AssertionError('Memory sensitivity numerical validation failed; inspect validation/memory_timescale.json.')
    return report


def render():
    from common import fig,letter,save,COL,ORDER,Line2D,plt,objective_display,objective_axis
    q=pd.read_csv(DATA/FILES[0]);sc=pd.read_csv(DATA/FILES[1]);q=q[q['mode'].eq('both')]
    f=fig(139);gs=f.add_gridspec(2,3,left=.092,right=.99,top=.965,bottom=.14,hspace=.49,wspace=.47)
    axes=[f.add_subplot(gs[i//3,i%3]) for i in range(6)]
    for ax,ex,key,ylabel in zip(axes[:4],[False,True,True,True],['behavior_h','D_h','V_h','gap_days1_7_circular_h'],['Behavioral period\nin vivo (h)','D period ex vivo (h)','V period ex vivo (h)','D–V phase gap,\nfirst 7 DD days (h)']):
        for c in ORDER:
            z=q[q.condition.eq(c)&q.ex_vivo.eq(ex)].sort_values('varied_tau_h')
            ax.plot(z.varied_tau_h,z[key],'-o',color=COL[c],ms=2.4,lw=.9)
            b=z[z.varied_tau_h.eq(240)].iloc[0]
            ax.scatter([240],[b[key]],s=18,c=[COL[c]],edgecolors='black',linewidths=.5,zorder=5)
        ax.set_ylabel(ylabel)
    for mode,ls,mark,label in [('both','-','o','D and X'),('D_only','--','s','D only'),('X_only',':','^','X only')]:
        z=sc[sc['mode'].eq(mode)].sort_values('varied_tau_h')
        axes[4].plot(z.varied_tau_h,objective_display(z.common_penalized_objective),color='#333333',ls=ls,marker=mark,ms=3,label=label)
    objective_axis(axes[4])
    axes[4].set_ylim(2.5,85);axes[4].set_yticks([3,10,30],['3','10','30'])
    axes[4].legend(frameon=False,fontsize=5.4,loc='lower left',bbox_to_anchor=(0,1.01),handlelength=1.6,labelspacing=.15,borderaxespad=0)
    contexts=[(False,'behavior_h','-','o','Behavior in vivo'),(True,'D_h','--','s','D ex vivo')]
    for c in ['T22','T26']:
        for ex,key,ls,mark,label in contexts:
            z=q[q.condition.eq(c)&q.ex_vivo.eq(ex)].sort_values('varied_tau_h')
            axes[5].plot(z.varied_tau_h,z[key],color=COL[c],ls=ls,marker=mark,ms=3)
            b=z[z.varied_tau_h.eq(240)].iloc[0]
            axes[5].scatter([240],[b[key]],s=18,c=[COL[c]],edgecolors='black',linewidths=.5,marker=mark,zorder=5)
    axes[5].set_ylabel('T22 and T26\nabsolute period (h)')
    axes[5].legend(handles=[Line2D([],[],color='#333333',ls=ls,marker=mark,ms=3,label=label) for ex,key,ls,mark,label in contexts],frameon=False,fontsize=5.4,loc='lower left',bbox_to_anchor=(0,1.01),handlelength=1.6,labelspacing=.15,borderaxespad=0)
    for i,ax in enumerate(axes):
        ax.axvline(240,color='#555555',ls='--',lw=.6,zorder=0)
        ax.set_xscale('log',base=2);ax.set_xlim(53,800);ax.set_xticks([60,120,240,480,720],['60','120','240','480','720'])
        ax.minorticks_off();ax.set_xlabel('Memory time constant (h)');letter(ax,chr(65+i),x=-.22,y=1.02)
    f.legend(handles=[Line2D([],[],color=COL[c],marker='o',ms=3,label=c) for c in ORDER],loc='lower center',bbox_to_anchor=(.52,.015),ncol=5,frameon=False)
    save(f,'Figure_S5',True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');parser.add_argument('--no-render',action='store_true');args=parser.parse_args()
    report=generate(args.force)
    if not args.no_render:render()
    print(json.dumps({k:report[k] for k in ['endpoint_rows','best_tested_shared_tau_h','maximum_period_difference_h','maximum_circular_gap_difference_h','passed']},indent=2))
