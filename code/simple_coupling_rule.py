"""Figure S6: the historical fixed-K_VD construction and its light-coupling map.

The historical qualitative example is displayed explicitly, without refitting
or altering the later independent-pair comparison. Three assigned conditions
do not identify a biological coupling law or validate unseen-schedule prediction.
"""
from pathlib import Path
import ast,hashlib,json,argparse
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import simple_coupling as model
from common import ROOT,DATA,TABLES,ORDER,COL,REG,fig,save,letter,Line2D,Patch

CONSTRUCTION='historical_fixed_K_VD'
TABLE_NAMES=['simple_coupling_rule_parameters.csv','simple_coupling_rule_endpoints.csv','simple_coupling_rule_diagnostics.csv']
FAMILIES={'photoperiod':('light_h',['SP','LD','LP']), 'T-cycle':('cycle_h',['T22','LD','T26'])}

def diagnostics(params):
 """Endpoint interpolation is a diagnostic, never a fit to all three points.

 The intermediate LD value is left out of the straight-line construction.
 Slopes and residuals are descriptive properties of prescribed coefficients.
 """
 rows=[]
 for family,(predictor,conditions) in FAMILIES.items():
  a=params.set_index('condition').loc[conditions];x=a[predictor].to_numpy(float)
  for coefficient,column in [('K_DV','K_DV_exvivo_rad_h'),('K_VD','K_VD_rad_h')]:
   y=a[column].to_numpy(float);slope=(y[-1]-y[0])/(x[-1]-x[0]);expected=y[0]+slope*(x-x[0])
   span=abs(y[-1]-y[0]);left=(y[1]-y[0])/(x[1]-x[0]);right=(y[2]-y[1])/(x[2]-x[1])
   for j,c in enumerate(conditions):
    rows.append(dict(construction=CONSTRUCTION,family=family,coefficient=coefficient,condition=c,predictor=predictor,predictor_h=x[j],actual_rad_h=y[j],endpoint_linear_rad_h=expected[j],residual_rad_h=y[j]-expected[j],left_slope_rad_h2=left,right_slope_rad_h2=right,endpoint_slope_rad_h2=slope,midpoint_relative_to_endpoint_span=abs(y[1]-expected[1])/span if span>1e-14 else 0.))
 return pd.DataFrame(rows)

def fingerprint():
 """Scientific fingerprint excludes figure layout and export-only code."""
 funcs={'initial','advance','trace','spectral_period','evaluate','contrasts'}
 base_tree=ast.parse(Path(model.__file__).read_text(encoding='utf-8'))
 science=[ast.dump(n,include_attributes=False) for n in base_tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in funcs]
 own_tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
 science += [ast.dump(n,include_attributes=False) for n in own_tree.body if isinstance(n,ast.FunctionDef) and n.name in {'diagnostics','calculate'}]
 configuration=dict(construction=CONSTRUCTION,order=ORDER,schedules=model.SCHEDULES.tolist(),dt=model.DT,families=FAMILIES)
 payload=json.dumps(configuration,sort_keys=True).encode()+''.join(science).encode()
 payload+=(DATA/'simple_coupling_input.json').read_bytes()+(DATA/'archived_targets.csv').read_bytes()
 return hashlib.sha256(payload).hexdigest()

def calculate():
 shared,pairs=model.initial();out,ends,traces=model.evaluate(shared,pairs,dt=model.DT,keep=True)
 _,half,_=model.evaluate(shared,pairs,dt=model.DT/2)
 source=json.loads((DATA/'simple_coupling_input.json').read_text(encoding='utf-8-sig'))
 dw=2*np.pi/shared[0]-2*np.pi/shared[1];params=[]
 for i,c in enumerate(ORDER):
  params.append(dict(construction=CONSTRUCTION,source_seed=source['historical_selected']['seed'],condition=c,cycle_h=model.SCHEDULES[i,0],light_h=model.SCHEDULES[i,1],K_DV_exvivo_rad_h=pairs[i,0],K_DV_intact_rad_h=pairs[i,0]+shared[2],K_VD_rad_h=pairs[i,1],epsilon_rad_h=shared[2],K_eff_exvivo_rad_h=pairs[i].sum(),K_eff_intact_rad_h=pairs[i].sum()+shared[2],delta_omega_rad_h=dw))
 params=pd.DataFrame(params);ends['construction']=CONSTRUCTION;traces['construction']=CONSTRUCTION
 rules=diagnostics(params);checks=[];convergence=[]
 for j,row in ends.iterrows():
  error=max(abs(row[k]-half.loc[j,k]) for k in ['D_h','V_h','behavior_h','release_gap_h','week_gap_h'])
  convergence.append(dict(condition=row.condition,context=row.context,maximum_endpoint_half_step_difference_h=float(error),passed=bool(error<.005)))
 for c in ORDER:
  a=traces[traces.condition.eq(c)&traces.context.eq('invivo')];b=traces[traces.condition.eq(c)&traces.context.eq('exvivo')]
  before=a.time_from_DD_h<0
  checks.append(dict(check=c+' identical pre-DD phases',passed=np.array_equal(a.loc[before,['phi_D','phi_V']].values,b.loc[b.time_from_DD_h<0,['phi_D','phi_V']].values)))
  checks.append(dict(check=c+' systemic offset removed only in explant DD',passed=bool(np.allclose(a.X_input,shared[2],rtol=0,atol=0) and np.allclose(b.loc[b.time_from_DD_h<0,'X_input'],shared[2],rtol=0,atol=0) and np.allclose(b.loc[b.time_from_DD_h>=0,'X_input'],0,rtol=0,atol=0))))
  for f in [a,b]:
   dd=f[f.time_from_DD_h>=0];delta=(dd.phi_D-dd.phi_V).to_numpy();k=float(dd.K_eff.iloc[0])
   sol=solve_ivp(lambda t,y:[dw-k*np.sin(y[0])],(0,336),[delta[0]],t_eval=dd.time_from_DD_h.to_numpy(),rtol=1e-11,atol=1e-12)
   error=float(np.max(abs(delta-sol.y[0])));checks.append(dict(check=c+' '+f.context.iloc[0]+' exact DD phase equation',error=error,passed=error<2e-5))
 checks.append(dict(check='all twelve qualitative spectral orderings',passed=bool((model.contrasts(out)>0).all())))
 checks.append(dict(check='shared K_VD',passed=bool(np.ptp(pairs[:,1])==0)))
 photo=rules[rules.family.eq('photoperiod')&rules.coefficient.eq('K_DV')&rules.condition.eq('LD')].iloc[0]
 cycle=rules[rules.family.eq('T-cycle')&rules.coefficient.eq('K_DV')&rules.condition.eq('LD')].iloc[0]
 report=dict(fingerprint=fingerprint(),construction=CONSTRUCTION,parameter_source='data/simple_coupling_input.json:historical_selected.vector',source_seed=source['historical_selected']['seed'],source_record_sha256=source['source_sha256'],refitted=False,independent_pair_comparison_replaced=False,shared_parameters=shared.tolist(),coupling_pairs=pairs.tolist(),checks=checks,step_halving=convergence,passed=all(r['passed'] for r in checks+convergence),all_spectral_orderings=bool((model.contrasts(out)>0).all()),minimum_spectral_contrast_h=float(model.contrasts(out).min()),behavior_RMSE_h=float(np.sqrt(np.mean((out[:,0,2]-model.BT)**2))),exvivo_regional_RMSE_h=float(np.sqrt(np.mean((out[:,1,:2]-model.ET[:,None])**2))),integration_step_h=model.DT,stored_step_h=.25,entrainment_h=720,DD_h=336,rule_diagnostics=dict(definition='interpolate a straight line between the two outer conditions, leaving the intermediate LD coefficient out; no coupling is re-estimated',photoperiod_K_DV_LD_residual_rad_h=float(photo.residual_rad_h),Tcycle_K_DV_LD_residual_rad_h=float(cycle.residual_rad_h),absolute_residual_ratio_Tcycle_to_photoperiod=float(abs(cycle.residual_rad_h/photo.residual_rad_h)),photoperiod_relative_to_endpoint_span=float(photo.midpoint_relative_to_endpoint_span),Tcycle_relative_to_endpoint_span=float(cycle.midpoint_relative_to_endpoint_span),interpretation='descriptive nonlinearity of this assigned example, not a uniquely identified response law or held-out prediction test'))
 return params,ends,rules,traces,report

def generate_rule(force=False):
 report_path=ROOT/'validation/simple_coupling_rule.json';cache_path=DATA/'simple_coupling_rule_results.json'
 if not force and report_path.exists() and cache_path.exists():
  report=json.loads(report_path.read_text())
  if report.get('fingerprint')==fingerprint() and report.get('output_hashes') and all((DATA/name).exists() and hashlib.sha256((DATA/name).read_bytes()).hexdigest()==value for name,value in report['output_hashes'].items()):
   cached=json.loads(cache_path.read_text())
   for name in TABLE_NAMES:pd.DataFrame(cached['tables'][name]).to_csv(TABLES/name,index=False)
   return report
 params,ends,rules,traces,report=calculate()
 tables=dict(zip(TABLE_NAMES,[params,ends,rules]))
 for name,frame in tables.items():frame.to_csv(TABLES/name,index=False)
 traces.to_csv(DATA/'simple_coupling_rule_traces.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 cache_path.write_text(json.dumps(dict(construction=CONSTRUCTION,tables={name:frame.to_dict('records') for name,frame in tables.items()}),indent=2),encoding='utf-8')
 report['output_hashes']={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['simple_coupling_rule_results.json','simple_coupling_rule_traces.csv.gz']}
 report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
 if not report['passed']:raise AssertionError(report)
 print('Historical coupling rule:',json.dumps(report['rule_diagnostics']),flush=True)
 return report

def render():
 from actogram_style import doubleplot_axis
 report=generate_rule();tr=pd.read_csv(DATA/'simple_coupling_rule_traces.csv.gz');tr['X']=np.nan
 cached=json.loads((DATA/'simple_coupling_rule_results.json').read_text());ends=pd.DataFrame(cached['tables'][TABLE_NAMES[1]]);rules=pd.DataFrame(cached['tables'][TABLE_NAMES[2]])
 f=fig(198);gs=f.add_gridspec(3,3,height_ratios=[1.05,1.05,.91],left=.09,right=.985,top=.968,bottom=.18,hspace=.38,wspace=.43)
 for row,ex in [(0,False),(1,True)]:
  nested=gs[row,:].subgridspec(1,5,wspace=.12)
  for j,c in enumerate(ORDER):
   ax=f.add_subplot(nested[0,j]);doubleplot_axis(ax,tr,c,ex,show_ylabels=j==0,show_xlabel=row==1,show_condition=row==0)
   if j==0:
    letter(ax,'AB'[row],x=-.4,y=1.02);ax.text(-.45,.5,'Ex vivo' if ex else 'In vivo',transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=7)
 ax=f.add_subplot(gs[2,0]);xx=np.arange(5)
 for i,c in enumerate(ORDER):
  inv=ends[ends.condition.eq(c)&ends.context.eq('invivo')].iloc[0];ex=ends[ends.condition.eq(c)&ends.context.eq('exvivo')].iloc[0]
  for dx,marker,value in [(-.22,'o',inv.behavior_h),(0,'s',ex.D_h),(.22,'^',ex.V_h)]:ax.scatter(i+dx,value,color=COL[c],marker=marker,s=16,zorder=3)
  ax.plot([i-.31,i-.13],[model.BT[i]]*2,color='black',lw=.9);ax.plot([i-.08,i+.3],[model.ET[i]]*2,color='black',lw=.9)
 ax.set(xticks=xx,xticklabels=ORDER,ylabel='DD period (h)',ylim=(22.6,25.05),xlim=(-.5,4.5));letter(ax,'C')
 ax.legend(handles=[Line2D([],[],marker=m,color='black',ls='',ms=2.6,label=l) for m,l in [('o','Behavior'),('s','D (ex)'),('^','V (ex)')]]+[Line2D([],[],color='black',label='Target')],frameon=False,fontsize=5.2,ncol=2,loc='upper left',handlelength=1,columnspacing=.5,handletextpad=.4,borderaxespad=.15)
 for col,family in [(1,'photoperiod'),(2,'T-cycle')]:
  ax=f.add_subplot(gs[2,col]);a=rules[rules.family.eq(family)]
  for coefficient,marker,ls in [('K_DV','o','-'),('K_VD','s','--')]:
   q=a[a.coefficient.eq(coefficient)].sort_values('predictor_h')
   ax.plot(q.predictor_h,q.actual_rad_h*1000,color='#444444',ls=ls,lw=.8,zorder=2)
   for _,r in q.iterrows():ax.scatter(r.predictor_h,r.actual_rad_h*1000,marker=marker,s=25,facecolors=COL[r.condition] if coefficient=='K_DV' else 'white',edgecolors=COL[r.condition],linewidths=1.1,zorder=5)
   if coefficient=='K_DV':
    ax.plot(q.predictor_h,q.endpoint_linear_rad_h*1000,color='#999999',ls=':',lw=1.1,zorder=3)
    ld=q[q.condition.eq('LD')].iloc[0]
    ax.plot([ld.predictor_h]*2,[ld.endpoint_linear_rad_h*1000,ld.actual_rad_h*1000],color='#777777',lw=.7,zorder=1)
    ax.text(.04 if family=='photoperiod' else .30,.045,f'LD deviation: +{ld.residual_rad_h*1000:.3f}',transform=ax.transAxes,fontsize=5.8,ha='left',va='bottom')
  xvals=sorted(a.predictor_h.unique());labels=['8\nSP','12\nLD','16\nLP'] if family=='photoperiod' else ['22\nT22','24\nLD','26\nT26']
  ax.set(xticks=xvals,xticklabels=labels,xlim=(min(xvals)-.6,max(xvals)+.6),ylim=(-17.5,8.5),ylabel='Coupling ($10^{-3}$ rad h$^{-1}$)',xlabel='Light duration (h); T = 24 h' if family=='photoperiod' else 'Cycle length (h); 50% light')
  letter(ax,'DE'[col-1])
 dynamic=[Line2D([],[],color=REG['D'],label='D'),Line2D([],[],color=REG['V'],label='V'),Line2D([],[],color='black',marker='o',ls='',ms=2.7,label='Behavior peak'),Patch(facecolor='#F6C85F',alpha=.35,label='Scheduled light')]
 f.legend(handles=dynamic,loc='lower center',bbox_to_anchor=(.53,.025),ncol=4,frameon=False,fontsize=5.7,handlelength=1.4,columnspacing=1.)
 rule_handles=[Line2D([],[],color='#444444',marker='o',ms=3,label='$K_{DV}$'),Line2D([],[],color='#444444',marker='s',mfc='white',ls='--',ms=3,label='$K_{VD}$'),Line2D([],[],color='#999999',ls=':',label='Endpoint line')]
 f.legend(handles=rule_handles,loc='lower center',bbox_to_anchor=(.65,.073),ncol=3,frameon=False,fontsize=5.6,handlelength=1.3,columnspacing=.8)
 save(f,'Figure_S6',supp=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');ap.add_argument('--render',action='store_true');args=ap.parse_args();generate_rule(args.force)
 if args.render:render()
