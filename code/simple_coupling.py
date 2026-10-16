"""Condition-specific fixed coupling construction and local sensitivity (Figure S6).

This deliberately descriptive comparator has two phase states and no memory,
systemic oscillator, or H state. Signed receiving-D coupling is attractive when
positive: dD/dt=wD+(K_DV[c]+X)*sin(V-D)+A_D*L*sin(theta-D).
dV/dt=wV+K_VD[c]*sin(D-V)+A_V*L*sin(theta-V). X=epsilon throughout
entrainment and intact DD; X=0 only in explant DD. Parameters for an unseen
schedule are not defined by this construction. It is not a budget-matched fit.
"""
from pathlib import Path
import os,json,hashlib,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from numba import njit,prange,set_num_threads
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from astropy.timeseries import LombScargle
from common import ROOT,DATA,TABLES,ORDER,COL,REG,fig,save,letter,schedule_legend,plt,Line2D,Patch

SCHEDULES=np.array([[24.,8.],[24.,16.],[24.,12.],[22.,11.],[26.,13.]])
TARGETS=pd.read_csv(DATA/'archived_targets.csv').set_index('condition').loc[ORDER]
BT=TARGETS.behavior_mean_h.to_numpy();ET=TARGETS.exvivo_scn_target_h.to_numpy()
DT=1/64;LOW=np.array([-.02,.0001]);HIGH=np.array([.005,.02])
TRACE_COLS=['time_from_DD_h','D','V','behavior','phi_D','phi_V','phi_output','K_DV_effective','K_VD','X_input','K_eff']
OUTPUTS=['simple_coupling_endpoints.csv','simple_coupling_parameters.csv','simple_coupling_shared_parameters.csv','simple_coupling_sensitivity.csv','simple_coupling_perturbation_summary.csv','simple_coupling_grid.csv','simple_coupling_convergence.csv','simple_coupling_search.csv']
TABLE_NAMES=OUTPUTS

def initial():
 q=json.loads((DATA/'simple_coupling_input.json').read_text(encoding='utf-8-sig'))
 v=np.array(q['historical_selected']['vector'])
 return v[[0,1,3,4,5]],np.column_stack([v[6:],np.repeat(v[2],5)])

@njit(cache=True)
def advance(d,v,shared,pair,t,cycle,photo,h,ex):
 dd=t>=720.-1e-9
 light=0. if dd else (1. if t%cycle<photo else 0.)
 theta=0. if dd else 2*np.pi*((t+h/2)%cycle)/cycle
 a=pair[0]+(0. if ex and dd else shared[2]);b=pair[1]
 fd=2*np.pi/shared[0]-a*np.sin(d-v)+shared[3]*light*np.sin(theta-d)
 fv=2*np.pi/shared[1]+b*np.sin(d-v)+shared[4]*light*np.sin(theta-v)
 md=d+h/2*fd;mv=v+h/2*fv
 fd=2*np.pi/shared[0]-a*np.sin(md-mv)+shared[3]*light*np.sin(theta-md)
 fv=2*np.pi/shared[1]+b*np.sin(md-mv)+shared[4]*light*np.sin(theta-mv)
 return d+h*fd,v+h*fv

@njit(cache=True)
def trace(shared,pair,i,dt,ex):
 cycle,photo=SCHEDULES[i];d=.35;v=0.
 stride=int(round(.25/dt));n=int(round(1056/dt))+1
 out=np.empty(((n-1)//stride+1,11));j=0
 for k in range(n):
  t=k*dt;x=0. if ex and t>=720-1e-9 else shared[2]
  read=np.arctan2(.8*np.sin(d)+.2*np.sin(v),.8*np.cos(d)+.2*np.cos(v))
  if k%stride==0:
   out[j]=np.array([t-720,-np.sin(d),-np.sin(v),-np.sin(read),d,v,read,pair[0]+x,pair[1],x,pair[0]+x+pair[1]]);j+=1
  d,v=advance(d,v,shared,pair,t,cycle,photo,dt,ex)
 return out[:j]

@njit(cache=True)
def summary_one(shared,pair,i,dt):
 cycle,photo=SCHEDULES[i];d=.35;v=0.
 for k in range(int(round(720/dt))):d,v=advance(d,v,shared,pair,k*dt,cycle,photo,dt,False)
 out=np.empty((2,3))
 for ex in range(2):
  dd=d;vv=v;sd=sv=st=stt=std=stv=sr=strd=0.;n=0
  for k in range(int(round(336/dt))+1):
   t=k*dt;sd+=dd;sv+=vv;st+=t;stt+=t*t;std+=t*dd;stv+=t*vv;n+=1
   read=dd+np.arctan2(.2*np.sin(vv-dd),.8+.2*np.cos(vv-dd));sr+=read;strd+=t*read
   dd,vv=advance(dd,vv,shared,pair,720+t,cycle,photo,dt,bool(ex))
  den=n*stt-st*st
  out[ex,0]=2*np.pi/((n*std-st*sd)/den)
  out[ex,1]=2*np.pi/((n*stv-st*sv)/den)
  out[ex,2]=2*np.pi/((n*strd-st*sr)/den)
 return out

@njit(cache=True)
def predict(shared,pairs,dt=.0625):
 out=np.empty((5,2,3))
 for i in range(5):out[i]=summary_one(shared,pairs[i],i,dt)
 return out

@njit(cache=True)
def contrasts(out):
 a=np.empty(12);a[0]=out[0,0,2]-out[2,0,2];a[1]=out[2,0,2]-out[1,0,2]
 a[2]=out[2,0,2]-out[3,0,2];a[3]=out[4,0,2]-out[2,0,2]
 for j in range(2):
  z=4+j*4;a[z]=out[0,1,j]-out[2,1,j];a[z+1]=out[2,1,j]-out[1,1,j]
  a[z+2]=out[3,1,j]-out[2,1,j];a[z+3]=out[2,1,j]-out[4,1,j]
 return a

@njit(cache=True)
def loss(out):
 if not np.isfinite(out).all():return 1e12
 score=0.
 for i in range(5):
  score+=((out[i,0,2]-BT[i])/.2)**2
  score+=.5*((out[i,1,0]-ET[i])/.25)**2+.5*((out[i,1,1]-ET[i])/.25)**2
 for v in contrasts(out):score+=20000.*max(0.,.025-v)**2
 return score

@njit(cache=True,parallel=True)
def evaluate_pairs(shared,candidates,i,baseline):
 out=np.empty((len(candidates),2,3));scores=np.empty(len(candidates))
 for j in prange(len(candidates)):
  out[j]=summary_one(shared,candidates[j],i,.0625)
  full=baseline.copy();full[i]=out[j];scores[j]=loss(full)
 return scores,out

def spectral_period(a,col):
 dd=a[a[:,0]>=0];ls=LombScargle(dd[:,0],dd[:,col],center_data=True,fit_mean=True)
 freqs,power=ls.autopower(minimum_frequency=1/40,maximum_frequency=1/12,samples_per_peak=12)
 j=int(np.argmax(power))
 if j in (0,len(freqs)-1):return float(1/freqs[j])
 opt=minimize_scalar(lambda x:-float(ls.power(x,method='cython')),bounds=(freqs[j-1],freqs[j+1]),method='bounded',options={'xatol':1e-12})
 return float(1/opt.x)

def evaluate(shared,pairs,dt=DT,keep=False):
 out=np.empty((5,2,3));rows=[];frames=[]
 dw=2*np.pi/shared[0]-2*np.pi/shared[1]
 for i,c in enumerate(ORDER):
  for ex in range(2):
   a=trace(shared,pairs[i],i,dt,bool(ex));dd=a[a[:,0]>=0];delta=dd[:,4]-dd[:,5]
   out[i,ex]=[spectral_period(a,j) for j in (1,2,3)]
   row=dict(condition=c,context='exvivo' if ex else 'invivo',D_h=out[i,ex,0],V_h=out[i,ex,1],behavior_h=out[i,ex,2],release_gap_h=np.angle(np.exp(1j*delta[0]))*24/(2*np.pi),week_gap_h=np.angle(np.exp(1j*delta[dd[:,0]<=168]).mean())*24/(2*np.pi),K_eff_rad_h=dd[0,10],delta_omega_rad_h=dw,locking_ratio=abs(dw/dd[0,10]))
   ld=a[(a[:,0]>=-168)&(a[:,0]<=0)];cycle=SCHEDULES[i,0]
   row['maximum_LD_period_error_h']=max(abs(2*np.pi/np.polyfit(ld[:,0],ld[:,j],1)[0]-cycle) for j in (4,5))
   rows.append(row)
   if keep:
    f=pd.DataFrame(a[a[:,0]>=-336],columns=TRACE_COLS);f['condition']=c;f['context']=row['context'];frames.append(f)
 return out,pd.DataFrame(rows),pd.concat(frames,ignore_index=True) if keep else None

def search(shared,start):
 """Equal deterministic coordinate-search budget for each of five pairs.

 One 13x13 bounded grid followed by four 7x7 neighborhoods per condition;
 all other conditions retain their current values. The same global loss is
 evaluated after replacing that condition's summary. Twenty-five checkpoints
 plus the historical warm start are re-evaluated spectrally before selection.
 """
 pairs=start.copy();baseline=predict(shared,pairs);checkpoints=[pairs.copy()];records=[]
 for stage,width in enumerate([None,.004,.001,.00025,.0000625]):
  for i,c in enumerate(ORDER):
   if width is None:xs=np.linspace(LOW[0],HIGH[0],13);ys=np.linspace(LOW[1],HIGH[1],13)
   else:
    xs=np.linspace(max(LOW[0],pairs[i,0]-width),min(HIGH[0],pairs[i,0]+width),7)
    ys=np.linspace(max(LOW[1],pairs[i,1]-width),min(HIGH[1],pairs[i,1]+width),7)
   candidates=np.vstack([pairs[i],np.array([(x,y) for x in xs for y in ys])])
   scores,outs=evaluate_pairs(shared,candidates,i,baseline);j=int(np.argmin(scores))
   pairs[i]=candidates[j];baseline[i]=outs[j];checkpoints.append(pairs.copy())
   records.append(dict(stage=stage,condition=c,evaluations=len(candidates),screening_score=float(scores[j]),K_DV_rad_h=pairs[i,0],K_VD_rad_h=pairs[i,1]))
  print('Simple coupling refinement stage',stage,'score',float(loss(baseline)),flush=True)
 fine=[]
 for j,q in enumerate(checkpoints):
  out,ends,_=evaluate(shared,q,dt=1/32)
  fine.append(dict(checkpoint=j,spectral_score=float(loss(out)),all_orderings=bool((contrasts(out)>0).all()),minimum_contrast_h=float(contrasts(out).min()),maximum_LD_period_error_h=float(ends.maximum_LD_period_error_h.max())))
 accepted=[r for r in fine if r['all_orderings'] and r['maximum_LD_period_error_h']<.02]
 if not accepted:raise AssertionError('No entrained qualitative construction, including historical warm start')
 selected=min(accepted,key=lambda r:r['spectral_score'])
 return checkpoints[selected['checkpoint']],records,fine,selected

@njit(cache=True,parallel=True)
def scan(shared,pairs,epsilon,all_pairs):
 out=np.empty((len(epsilon),14))
 for j in prange(len(epsilon)):
  s=shared.copy();s[2]=epsilon[j];z=predict(s,all_pairs[j]);cc=contrasts(z)
  out[j,:12]=cc;out[j,12]=loss(z);out[j,13]=z[4,1,0]-z[3,1,0]
 return out

def fingerprint():
 return hashlib.sha256(Path(__file__).read_bytes()+(DATA/'simple_coupling_input.json').read_bytes()+(DATA/'archived_targets.csv').read_bytes()).hexdigest()

def generate(force=False):
 report_path=ROOT/'validation/simple_coupling.json';cache=DATA/'simple_coupling_results.json'
 if not force and report_path.exists() and cache.exists():
  old=json.loads(report_path.read_text());saved=json.loads(cache.read_text())
  if old.get('fingerprint')==fingerprint() and all((DATA/n).exists() and hashlib.sha256((DATA/n).read_bytes()).hexdigest()==sha for n,sha in old.get('output_hashes',{}).items()):
   for name in OUTPUTS:pd.DataFrame(saved['tables'][name]).to_csv(TABLES/name,index=False)
   return old
 set_num_threads(min(6,os.cpu_count() or 1));shared,start=initial()
 pairs,search_rows,fine,selected=search(shared,start)
 out,ends,traces=evaluate(shared,pairs,keep=True)
 _,half,_=evaluate(shared,pairs,dt=DT/2)
 conv=ends[['condition','context']].copy()
 for key in ['D_h','V_h','behavior_h','release_gap_h','week_gap_h']:conv[key+'_half_step_error']=half[key]-ends[key]
 traces.to_csv(DATA/'simple_coupling_traces.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 grid_meta=[];grid_pairs=[];eps=[]
 for x in np.linspace(-.01,.01,41):
  for y in np.linspace(0,.02,41):
   q=pairs.copy();q[:,0]+=x;grid_pairs.append(q);eps.append(y);grid_meta.append(dict(K_DV_common_offset_rad_h=x,epsilon_rad_h=y))
 z=scan(shared,pairs,np.array(eps),np.array(grid_pairs))
 for r,v in zip(grid_meta,z):r.update(all_orderings=bool((v[:12]>0).all()),minimum_contrast_h=float(v[:12].min()),objective=float(v[12]))
 rng=np.random.default_rng(220926);radii=[.01,.02,.05,.1,.2];meta=[];qs=[];eps=[]
 for radius in radii:
  for j in range(512):
   perturb=rng.uniform(-radius,radius,11)
   qs.append(pairs+perturb[:10].reshape(5,2)*np.maximum(abs(pairs),shared[2]));eps.append(shared[2]*(1+perturb[10]));meta.append(dict(relative_radius=radius,replicate=j))
 z=scan(shared,pairs,np.array(eps),np.array(qs));spectral_checks=[]
 for r,v in zip(meta,z):r.update(all_orderings=bool((v[:12]>0).all()),minimum_contrast_h=float(v[:12].min()),all_margin_0_01h=bool((v[:12]>.01).all()))
 for j,r in enumerate(meta):
  if r['replicate']>=8:continue
  s=shared.copy();s[2]=eps[j];spec,_,_=evaluate(s,qs[j],dt=1/32);cc=contrasts(spec)
  spectral_checks.append(dict(relative_radius=r['relative_radius'],replicate=r['replicate'],regression_orderings=r['all_orderings'],spectral_orderings=bool((cc>0).all()),spectral_minimum_contrast_h=float(cc.min())))
 sensitivity=pd.DataFrame(meta);agg=sensitivity.groupby('relative_radius').agg(retained=('all_orderings','sum'),total=('all_orderings','size'),retention_fraction=('all_orderings','mean'),margin_0_01h_fraction=('all_margin_0_01h','mean')).reset_index()
 spec_df=pd.DataFrame(spectral_checks)
 agg=agg.merge(spec_df.groupby('relative_radius').agg(spectral_checked=('replicate','size'),spectral_retained=('spectral_orderings','sum'),spectral_retention_fraction=('spectral_orderings','mean')).reset_index())
 params=[]
 dw=2*np.pi/shared[0]-2*np.pi/shared[1]
 for i,c in enumerate(ORDER):
  params.append(dict(condition=c,cycle_h=SCHEDULES[i,0],light_h=SCHEDULES[i,1],K_DV_exvivo_rad_h=pairs[i,0],K_DV_intact_rad_h=pairs[i,0]+shared[2],K_VD_rad_h=pairs[i,1],epsilon_rad_h=shared[2],K_eff_exvivo_rad_h=pairs[i].sum(),K_eff_intact_rad_h=pairs[i].sum()+shared[2],delta_omega_rad_h=dw,historical_K_DV_rad_h=start[i,0],historical_K_VD_rad_h=start[i,1],K_DV_lower_bound_rad_h=LOW[0],K_DV_upper_bound_rad_h=HIGH[0],K_VD_lower_bound_rad_h=LOW[1],K_VD_upper_bound_rad_h=HIGH[1]))
 shared_table=pd.DataFrame(dict(parameter=['tau_D','tau_V','epsilon','A_D','A_V','behavior_D_weight','behavior_V_weight','initial_phi_D','initial_phi_V'],value=list(shared)+[.8,.2,.35,0],unit=['h','h','rad/h','rad/h','rad/h','dimensionless','dimensionless','rad','rad'],status=['Fixed historical construction']*5+['Fixed readout']*2+['Fixed initial state']*2))
 tables=dict(zip(OUTPUTS,[ends,pd.DataFrame(params),shared_table,sensitivity,agg,pd.DataFrame(grid_meta),conv,pd.DataFrame(search_rows)]))
 checks=[]
 for c in ORDER:
  a=traces[traces.condition.eq(c)&traces.context.eq('invivo')];b=traces[traces.condition.eq(c)&traces.context.eq('exvivo')]
  pre=a.time_from_DD_h<0
  checks.append(dict(check=c+' identical pre-DD histories',passed=np.array_equal(a.loc[pre,['phi_D','phi_V']].values,b.loc[b.time_from_DD_h<0,['phi_D','phi_V']].values)))
  for ex,f in [(False,a),(True,b)]:
   dd=f[f.time_from_DD_h>=0];delta=(dd.phi_D-dd.phi_V).to_numpy();k=float(dd.K_eff.iloc[0])
   sol=solve_ivp(lambda t,y:[dw-k*np.sin(y[0])],(0,336),[delta[0]],t_eval=dd.time_from_DD_h.to_numpy(),rtol=1e-11,atol=1e-12)
   err=float(np.max(abs(delta-sol.y[0])));checks.append(dict(check=c+str(ex)+' exact DD relative-phase equation',error=err,passed=err<2e-5))
 maxconv=float(abs(conv.iloc[:,2:]).max().max());checks.append(dict(check='step halving',error=maxconv,passed=maxconv<.005))
 zero=shared.copy();zero[2]=0.
 for i,c in enumerate(ORDER):
  a=trace(zero,pairs[i],i,DT,False);b=trace(zero,pairs[i],i,DT,True)
  checks.append(dict(check=c+' zero-offset context identity',passed=np.array_equal(a,b)))
 for name,frame in tables.items():frame.to_csv(TABLES/name,index=False)
 cache.write_text(json.dumps(dict(tables={name:frame.to_dict('records') for name,frame in tables.items()},spectral_validation=spectral_checks,finalists=fine,shared=shared.tolist(),couplings=pairs.tolist()),indent=2),encoding='utf-8')
 report=dict(fingerprint=fingerprint(),passed=all(x['passed'] for x in checks),checks=checks,model='two_fixed_period_oscillators_condition_specific_signed_couplings',shared_parameters=shared.tolist(),coupling_pairs=pairs.tolist(),selected=selected,all_spectral_orderings=bool((contrasts(out)>0).all()),minimum_spectral_contrast_h=float(contrasts(out).min()),behavior_RMSE_h=float(np.sqrt(np.mean((out[:,0,2]-BT)**2))),exvivo_regional_RMSE_h=float(np.sqrt(np.mean((out[:,1,:2]-ET[:,None])**2))),grid_retained=int(sum(r['all_orderings'] for r in grid_meta)),grid_total=len(grid_meta),perturbation_summary=agg.to_dict('records'),spectral_validation_agreements=int(sum(r['regression_orderings']==r['spectral_orderings'] for r in spectral_checks)),spectral_validation_n=len(spectral_checks),search={'method':'deterministic coordinate grids, five passes, same budget per schedule','bounds':{'K_DV':LOW[[0]].tolist()+HIGH[[0]].tolist(),'K_VD':LOW[[1]].tolist()+HIGH[[1]].tolist()},'evaluations_per_condition':370,'selection':'lowest refined spectral objective among all-12-ordering checkpoints with maximum entrainment period error <0.02 h','objective':'sum((behavior-target)/0.2)^2 + 0.5*sum_{D,V}((ex-target)/0.25)^2 + 20000*sum(max(0,0.025-ordering_contrast)^2)','matched_to_main_optimizer':False},sensitivity={'seed':220926,'n_per_radius':512,'perturbed_parameters':'all ten independently assigned K values and common epsilon','scale':'independent uniform +/-radius*max(abs(K),epsilon); epsilon +/-radius*epsilon','criterion':'all twelve contrasts >0; behavior: SP>LD>LP and T26>LD>T22; ex vivo D and V: SP>LD>LP and T22>LD>T26','grid':'common additive offset to five K_DV versus common epsilon; both LD and DD rerun','interpretation':'local neighborhood of selected construction, not a global feasible-volume estimate or a matched robustness/generalization comparison'},output_hashes={n:hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in ['simple_coupling_results.json','simple_coupling_traces.csv.gz']})
 report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
 if not report['passed']:raise AssertionError(report)
 print('Simple coupling analysis',json.dumps({k:report[k] for k in ['all_spectral_orderings','minimum_spectral_contrast_h','behavior_RMSE_h','exvivo_regional_RMSE_h','grid_retained','grid_total']}),flush=True)
 return report

def render():
 from actogram_style import doubleplot_axis
 report=generate();cache=json.loads((DATA/'simple_coupling_results.json').read_text());tables={k:pd.DataFrame(v) for k,v in cache['tables'].items()}
 tr=pd.read_csv(DATA/'simple_coupling_traces.csv.gz');tr['X']=np.nan
 f=fig(232);gs=f.add_gridspec(4,3,height_ratios=[1.05,1.05,.95,.9],left=.09,right=.985,top=.97,bottom=.09,hspace=.38,wspace=.45)
 for row,ex in [(0,False),(1,True)]:
  nested=gs[row,:].subgridspec(1,5,wspace=.12)
  for j,c in enumerate(ORDER):
   ax=f.add_subplot(nested[0,j]);doubleplot_axis(ax,tr,c,ex,show_ylabels=j==0,show_xlabel=row==1,show_condition=row==0)
   if j==0:
    letter(ax,'AB'[row],x=-.4,y=1.02);ax.text(-.45,.5,'Ex vivo' if ex else 'In vivo',transform=ax.transAxes,rotation=90,va='center',ha='center',fontsize=7)
 ax=f.add_subplot(gs[2,0]);params=tables['simple_coupling_parameters.csv'];xx=np.arange(5)
 ax.axhline(0,color='#777777',ls='--',lw=.6)
 for j,c in enumerate(ORDER):
  q=params[params.condition.eq(c)].iloc[0]
  ax.plot([j-.12,j-.12],[q.K_DV_exvivo_rad_h*1000,q.K_DV_intact_rad_h*1000],color=COL[c],lw=.7)
  ax.scatter([j-.12,j-.12,j+.12],[q.K_DV_exvivo_rad_h*1000,q.K_DV_intact_rad_h*1000,q.K_VD_rad_h*1000],marker='o',s=18,facecolors=[COL[c],'white',COL[c]],edgecolors=COL[c])
 ax.set(xticks=xx,xticklabels=ORDER,ylabel='Signed coupling\n($10^{-3}$ rad h$^{-1}$)');letter(ax,'C')
 ax.legend(handles=[Line2D([],[],marker='o',color='black',ls='',ms=3,label='$K_{DV}$ explant'),Line2D([],[],marker='o',mfc='white',color='black',ls='',ms=3,label=r'$K_{DV}+\epsilon$ intact')],frameon=False,fontsize=5.5,loc='lower left')
 ax.text(.97,.95,'Right: $K_{VD}$',transform=ax.transAxes,ha='right',va='top',fontsize=6)
 ax=f.add_subplot(gs[2,1:]);ends=tables['simple_coupling_endpoints.csv']
 for i,c in enumerate(ORDER):
  inv=ends[ends.condition.eq(c)&ends.context.eq('invivo')].iloc[0];ex=ends[ends.condition.eq(c)&ends.context.eq('exvivo')].iloc[0]
  for dx,key,marker,value in [(-.22,'Behavior','o',inv.behavior_h),(0,'Ex vivo D','s',ex.D_h),(.22,'Ex vivo V','^',ex.V_h)]:ax.scatter(i+dx,value,color=COL[c],marker=marker,s=24,zorder=3)
  ax.plot([i-.31,i-.13],[BT[i]]*2,color='black',lw=1);ax.plot([i-.08,i+.3],[ET[i]]*2,color='black',lw=1)
 ax.set(xticks=xx,xticklabels=ORDER,ylabel='DD period (h)');letter(ax,'D',x=-.09)
 ax.legend(handles=[Line2D([],[],marker=m,color='black',ls='',ms=3,label=l) for m,l in [('o','In vivo behavior'),('s','Ex vivo D'),('^','Ex vivo V')]]+[Line2D([],[],color='black',label='Target')],frameon=False,fontsize=5.6,ncol=2,loc='upper left')
 ax=f.add_subplot(gs[3,:2]);grid=tables['simple_coupling_grid.csv'];pivot=grid.pivot(index='epsilon_rad_h',columns='K_DV_common_offset_rad_h',values='all_orderings')
 ax.pcolormesh(pivot.columns.to_numpy()*1000,pivot.index.to_numpy()*1000,pivot.to_numpy().astype(int),cmap='Greys',vmin=0,vmax=1,shading='nearest',rasterized=True)
 ax.scatter(0,report['shared_parameters'][2]*1000,marker='*',s=75,facecolors='#E69F00',edgecolors='black',lw=.5,zorder=4)
 ax.set(xlabel='Common offset to $K_{DV}$ ($10^{-3}$ rad h$^{-1}$)',ylabel=r'$\epsilon$ ($10^{-3}$ rad h$^{-1}$)');letter(ax,'E',x=-.09)
 ax.text(.02,.96,'Black: all period orderings retained',transform=ax.transAxes,ha='left',va='top',fontsize=6,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
 ax=f.add_subplot(gs[3,2]);agg=tables['simple_coupling_perturbation_summary.csv']
 ax.plot(agg.relative_radius*100,agg.retention_fraction*100,'o-',color='#333333',ms=3)
 ax.set(xlabel='Joint perturbation (%)',ylabel='Orderings retained (%)',ylim=(-3,103),xticks=[1,5,10,20]);letter(ax,'F')
 ax.text(.97,.95,'512 draws / scale',transform=ax.transAxes,ha='right',va='top',fontsize=6)
 ax.text(.02,.05,'Full-model phase regression',transform=ax.transAxes,ha='left',va='bottom',fontsize=6)
 handles=[Line2D([],[],color=REG['D'],label='D'),Line2D([],[],color=REG['V'],label='V'),Line2D([],[],color='black',marker='o',ls='',ms=2.7,label='Behavior peak'),Patch(facecolor='#F6C85F',alpha=.35,label='Scheduled light')]
 f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.025),ncol=4,frameon=False,fontsize=6)
 save(f,'Figure_S6',supp=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');ap.add_argument('--render',action='store_true');args=ap.parse_args();generate(args.force)
 if args.render:render()
