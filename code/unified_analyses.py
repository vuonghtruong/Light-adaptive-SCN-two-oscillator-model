"""Reintegrate every retained reference-model analysis using one selected fit.

The seed-1729 initial phase triples and intervention definitions are unchanged.
Existing derived data are reused only when their parameter/code fingerprint and
content hashes agree. Literature extraction and optimization are separate inputs.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, time
import numpy as np
import pandas as pd
from numba import njit
from astropy.timeseries import LombScargle
import reference_model as m

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; VALIDATION=ROOT/'validation'
DT=.015625
FUTURE=['18:6 LD','T23','T25']
GROUP_FILES={
 'ensembles':['perturbation_initial_ensemble.csv','original_logic_prediction_ensemble.csv'],
 'explant':['explant_controls.csv','explant_coupling_sweep.csv','explant_step_halving.csv'],
 'weighting':['perturbation_weighting.csv'],
 'memory':['perturbation_memory_contrasts.csv'],
 'diagnostics':['perturbation_reference.csv','original_logic_predictions.csv','reference_frozen_locking.csv','reference_frozen_phase_flow.csv','reference_long_DD.csv','original_logic_phase_reorganization.csv','original_logic_bifurcation_branches.csv'],
 'reviewer':['regime_map.csv','T22_additive_trace.csv','T22_additive_spectrum.csv','fixed_period_release.csv','natural_vs_network_windows.csv'],
 'landscape':['original_parameter_landscape.csv'],
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint():
    h=hashlib.sha256(json.dumps(m.PARAMS,sort_keys=True).encode())
    for name in ['unified_analyses.py','reference_model.py','fit_protocol.py','architecture_model.py']:
        h.update((ROOT/'code'/name).read_bytes())
    return h.hexdigest()
def table(name,rows):
    data=pd.DataFrame(rows);data.to_csv(DATA/name,index=False);return data

def measures(a):
    z=m.metrics(a);ld=a[a[:,0]<0];dd=a[a[:,0]>=0]
    for region,col in [('D',10),('X',11)]:
        z[f'tau_{region}_entrainment_min_h']=float(ld[:,col].min())
        z[f'tau_{region}_entrainment_max_h']=float(ld[:,col].max())
        z[f'tau_{region}_entrainment_clipped_fraction']=float(np.mean((ld[:,col]<=16+1e-8)|(ld[:,col]>=34-1e-8)))
    for region,col in [('D',5),('V',6),('X',7)]:
        z[f'{region}_DD_slope_period_h']=float(2*np.pi/np.polyfit(dd[:,0],np.unwrap(dd[:,col]),1)[0])
    return z

@njit(cache=True)
def explant_integrate(p,cycle,photo,dt,cscale,mret,kick_h):
    """Same Euler order as reference_model; reset MD only at DD release."""
    release=720.;n=int(round(1056/dt))+1
    out=np.zeros((n,21));d=.35;v=0.;x=-.2;md=0.;mx=0.;twopi=2*np.pi;changed=False
    for i in range(n):
        t=i*dt;isdd=t>=release-1e-9
        if isdd and not changed:
            d=(d+kick_h*twopi/24)%twopi;md*=mret;changed=True
        light=0. if isdd else (1. if t%cycle<photo else 0.)
        theta=0. if isdd else twopi*(t%cycle)/cycle
        drd=p[5]*light*np.sin(theta-d)+p[6]*((1+np.sin(d))/2-light)
        drx=p[7]*light*np.sin(theta-x)+p[8]*((1+np.sin(x))/2-light)
        if isdd:drd=0.;drx=0.
        md+=dt*(drd-md)/p[17];mx+=dt*(drx-mx)/p[18]
        td=min(34.,max(16.,p[0]-p[3]*np.tanh(md)))
        tx=min(34.,max(16.,p[2]-p[4]*np.tanh(mx)))
        scale=cscale if isdd else 1.;kvd=p[9]*scale;kdv=p[10]*scale
        sr=.8*np.sin(d)+(1-.8)*np.sin(v);cr=.8*np.cos(d)+(1-.8)*np.cos(v);read=np.arctan2(sr,cr)
        factor=0. if isdd else 1.
        fxd=factor*p[11]*np.sin(x-d);fxv=factor*p[12]*np.sin(x-v)
        gap=((d-v+np.pi)%twopi-np.pi)*24/twopi
        folded=gap-12 if gap>6 else (gap+12 if gap< -6 else gap)
        out[i]=np.array([t-release,-np.sin(d),-np.sin(v),-np.sin(x),-np.sin(read),d,v,x,md,mx,td,tx,gap,folded,fxd,fxv,kvd-kdv,np.sqrt(sr*sr+cr*cr),read,drd,drx])
        dd=twopi/td-kdv*np.sin(v-d+p[14])+fxd+p[15]*light*np.sin(theta-d)
        dv=twopi/p[1]+kvd*np.sin(d-v-p[14])+fxv+p[16]*p[15]*light*np.sin(theta-v)
        dx=twopi/tx+p[13]*np.sin(read-x)
        d=(d+dt*dd)%twopi;v=(v+dt*dv)%twopi;x=(x+dt*dx)%twopi
    return out

def ensembles():
    starts=np.random.default_rng(1729).uniform(-np.pi,np.pi,(64,3));old=[];future=[]
    for sid,initial in enumerate(starts):
        for c in m.ORDER+FUTURE:
            for ex in [False,True]:
                a=m.simulate(c,ex,dt=.0625,initial=initial)
                r=dict(start_id=sid,condition=c,ex_vivo=ex,dt_h=.0625,initial_D=initial[0],initial_V=initial[1],initial_X=initial[2],**measures(a))
                if c in FUTURE:r['X_h']=np.nan if ex else m.period(a,3)
                (old if c in m.ORDER else future).append(r)
        if (sid+1)%16==0:print(f'Unified phase ensemble {sid+1}/64',flush=True)
    table(GROUP_FILES['ensembles'][0],old);table(GROUP_FILES['ensembles'][1],future)
    return dict(seed=1729,initial_triples=64,reference_rows=len(old),prediction_rows=len(future),integration_step_h=.0625)

def explant():
    p=np.array([m.PARAMS[k] for k in m.KEYS]);checks=[];rows=[];cv=[];coupling=[]
    for c in ['LD','T22','T26']:
        a=explant_integrate(p,*m.SCHEDULES[c],DT,1.,1.,0.)
        b=m.simulate(c,True,dt=DT)
        error=float(abs(a-b).max());assert error<1e-10
        checks.append(dict(condition=c,unperturbed_maximum_state_error=error))
    for i,(cs,mr,ph) in enumerate(itertools.product([.5,1.,1.5],[0.,.5,1.],[-2.,0.,2.])):
        for c in ['LD','T22','T26']:
            a=explant_integrate(p,*m.SCHEDULES[c],DT,cs,mr,ph)
            b=explant_integrate(p,*m.SCHEDULES[c],DT/2,cs,mr,ph)
            z=measures(a);fine=m.metrics(b)
            meta=dict(condition=c,X_feedback_retained=0.,coupling_scale=cs,memory_D_retained=mr,D_phase_advance_h=ph,memory_X_retained=1.,dt_h=DT)
            rows.append(dict(**meta,**z))
            for key in ['behavior_h','D_h','V_h']:
                cv.append(dict(**meta,metric=key,absolute_difference_h=abs(z[key]-fine[key])))
        if (i+1)%9==0:print(f'Unified explant combinations {i+1}/27',flush=True)
    for c in ['LD','T22','T26']:
        for cs in [.5,.75,1.,1.25,1.5]:
            a=m.simulate(c,True,dt=DT,cscale=cs)
            coupling.append(dict(condition=c,coupling_scale=cs,**m.metrics(a)))
    table('explant_controls.csv',rows);table('explant_step_halving.csv',cv);table('explant_coupling_sweep.csv',coupling)
    return dict(integration_step_h=DT,fine_step_h=DT/2,cases=len(rows),period_checks=len(cv),maximum_period_difference_h=max(r['absolute_difference_h'] for r in cv),implementation_checks=checks)

def weighting():
    rows=[]
    for w in [.5,.7,.8,.9,1.]:
        for branch in ['readout_only','readout_and_feedback']:
            for c in m.ORDER:
                a=m.simulate(c,dt=DT,wr=w,wf=.8 if branch=='readout_only' else w)
                rows.append(dict(condition=c,D_weight=w,branch=branch,dt_h=DT,**measures(a)))
    table('perturbation_weighting.csv',rows)
    return dict(cases=len(rows),integration_step_h=DT)

def memory():
    tests={'reference':{},'D_memory_off':{'g_d':0.},'X_memory_off':{'g_x':0.},'D_and_X_memory_off':{'g_d':0.,'g_x':0.},'phase_drive_off':{'gphi_d':0.,'gphi_x':0.},'photoperiod_drive_off':{'gp_d':0.,'gp_x':0.}}
    for scale in [.5,1.5]:
        for label,keys in [('phase_gain',['gphi_d','gphi_x']),('photoperiod_gain',['gp_d','gp_x']),('period_gain',['g_d','g_x'])]:
            tests[f'{label}_{scale:g}x']={k:m.PARAMS[k]*scale for k in keys}
    for tau in [120.,480.]:tests[f'memory_tau_{tau:g}h']={'mem_tau_d':tau,'mem_tau_x':tau}
    rows=[]
    for name,changes in tests.items():
        for ex in [False,True]:
            values={c:m.metrics(m.simulate(c,ex,dt=DT,changes=changes)) for c in m.ORDER}
            for c in ['T22','T26']:
                diff={f'{key}_minus_LD_h':values[c][key]-values['LD'][key] for key in ['behavior_h','D_h','V_h','gap_first_peak_h','gap_days1_7_mean_h']}
                rows.append(dict(test=name,ex_vivo=ex,condition=c,**diff))
    table('perturbation_memory_contrasts.csv',rows)
    return dict(test_count=len(tests),conditions=len(m.ORDER),integration_step_h=DT,alternative_memory_timescales_h=[120.,480.])

def diagnostics():
    refs=[];pred=[];locks=[];flows=[];relax=[];phase=[];branches=[]
    for c in m.ORDER+FUTURE:
        for ex in [False,True]:
            a=m.simulate(c,ex,dt=DT);z=measures(a)
            refs.append(dict(condition=c,ex_vivo=ex,dt_h=DT,**z))
            pred.append(dict(condition=c,ex_vivo=ex,dt_h=DT,**m.metrics(a),X_h=np.nan if ex else m.period(a,3)))
            if not ex:continue
            release=a[np.flatnonzero(a[:,0]>=0)[0]];dw=2*np.pi/release[10]-2*np.pi/m.PARAMS['tau_v'];ke=m.PARAMS['kvd']-m.PARAMS['kdv']
            if c in m.ORDER:
                locks.append(dict(condition=c,DD_time_h=0.,tau_D_at_release_h=release[10],tau_V_h=m.PARAMS['tau_v'],delta_omega_rad_h=dw,K_eff_rad_h=ke,frozen_ratio=dw/ke,has_frozen_fixed_point=abs(dw/ke)<=1))
                flows.extend(dict(condition=c,delta_rad=delta,phase_gap_derivative_rad_h=dw-ke*np.sin(delta)) for delta in np.linspace(-np.pi,np.pi,301))
            dd=a[a[:,0]>=0][::16];delta=np.unwrap(dd[:,5])-np.unwrap(dd[:,6]);delta0=delta[0]
            for r,de in zip(dd,delta):
                instantaneous=2*np.pi/r[10]-2*np.pi/m.PARAMS['tau_v'];flow=instantaneous-ke*np.sin(de)
                phase.append(dict(condition=c,DD_h=r[0],tau_D_h=r[10],delta_omega_rad_h=instantaneous,K_eff_rad_h=ke,instantaneous_frozen_ratio=abs(instantaneous/ke),unwrapped_D_minus_V_rad=de,phase_change_from_release_h=(de-delta0)*24/(2*np.pi),phase_flow_rad_h=flow,phase_velocity_h_per_day=flow*24*24/(2*np.pi)))
            dwmean=2*np.pi/z['tau_D_mean_first7d_h']-2*np.pi/m.PARAMS['tau_v'];kdv=m.PARAMS['kdv'];lower=kdv-abs(dwmean);upper=kdv+abs(dwmean)
            for lo,hi in [(0,max(0,lower)),(min(.05,upper),.05)]:
                if hi<=lo:continue
                for kvd in np.linspace(lo,hi,300):
                    k=kvd-kdv
                    if abs(k)<1e-12 or abs(dwmean/k)>1:continue
                    angle=np.arcsin(dwmean/k)
                    for delta in [angle,(np.pi-angle+np.pi)%(2*np.pi)-np.pi]:
                        branches.append(dict(condition=c,K_VD_rad_h=kvd,K_DV_rad_h=kdv,delta_omega_rad_h=dwmean,phase_rad=delta,stability='stable' if k*np.cos(delta)>0 else 'unstable',frozen_parameter_definition='2pi / mean(tau_D over DD0-168h) minus omegaV'))
        if c in m.ORDER:
            a=m.simulate(c,dt=DT,ddh=1440.)
            for day in range(0,60,5):relax.append(dict(condition=c,window_start_day=day,window_end_day=day+5,behavior_h=m.period(a,4,start=24*day,end=24*(day+5))))
    for name,rows in zip(GROUP_FILES['diagnostics'],[refs,pred,locks,flows,relax,phase,branches]):table(name,rows)
    return dict(integration_step_h=DT,long_DD_days=60,period_window_days=5,bifurcation_definition='frozen first-seven-day mean natural D period',branches=len(branches))

def classify(z):
    both=z['D_entrained_frequency'] and z['V_entrained_frequency']
    slip=abs(z['relative_phase_net_drift_cycles'])>=.25 or z['relative_phase_range_cycles']>=.5
    locked=not slip and abs(z['D_minus_V_slope_period_h'])<.1 and z['DV_locking_resultant']>=.9
    if both and locked:return 0
    if locked:return 1
    if z['D_entrained_frequency'] or z['V_entrained_frequency']:return 2
    return 3

def reviewer():
    rows=[];candidates=[]
    for T in range(20,29):
        name=f'T{T}';m.SCHEDULES[name]=(float(T),T/2)
        for scale in [0,.25,.5,.75,1,1.25,1.5,2]:
            a=m.simulate(name,dt=.03125,cscale_before=scale,cscale=scale)
            z=m.locking(a,-336,0,T);rows.append(dict(T_h=T,coupling_scale=scale,state=classify(z),**z))
            if T==22:candidates.append((z['relative_phase_range_cycles'],scale,a))
    regime=table('regime_map.csv',rows)
    _,scale,a=max(candidates,key=lambda r:r[0]);representative_scale=scale;g=a[(a[:,0]>=-336)&(a[:,0]<=0)][::8]
    y=.8*np.sin(g[:,5])+.2*np.sin(g[:,6]);table('T22_additive_trace.csv',dict(time_h=g[:,0],D=np.sin(g[:,5]),V=np.sin(g[:,6]),additive=y,coupling_scale=np.full(len(g),scale)))
    f=np.linspace(1/30,1/18,2000);sp=[]
    for key,val in [('D',np.sin(g[:,5])),('V',np.sin(g[:,6])),('additive',y)]:
        power=LombScargle(g[:,0],val).power(f);sp.extend(dict(period_h=1/x,signal=key,power=p) for x,p in zip(f,power))
    table('T22_additive_spectrum.csv',sp)
    rows=[]
    for T in [22,26]:
        for scale in [0,.25,.5,.75,1,1.25,1.5,2]:
            a=m.simulate(f'T{T}',True,dt=DT,cscale_before=scale,cscale=1,changes={'g_d':0.,'g_x':0.});z=m.locking(a,-336,0,T)
            for start,end,lab in [(0,168,'DD1–7'),(0,336,'DD1–14'),(168,336,'DD7–14')]:
                rows.append(dict(T_h=T,coupling_scale_before=scale,window=lab,phase_range_cycles=z['relative_phase_range_cycles'],phase_drift_cycles=z['relative_phase_net_drift_cycles'],D_h=m.period(a,1,start,end),V_h=m.period(a,2,start,end),network_h=m.period(a,4,start,end),tau_D_h=m.PARAMS['tau_d0'],tau_V_h=m.PARAMS['tau_v']))
    table('fixed_period_release.csv',rows);rows=[]
    for c in m.ORDER:
        a=m.simulate(c,True,dt=DT)
        for start,end,lab in [(0,168,'DD1–7'),(0,336,'DD1–14'),(168,336,'DD7–14')]:
            g=a[(a[:,0]>=start)&(a[:,0]<=end)]
            rows.append(dict(condition=c,window=lab,natural_D_h=g[:,10].mean(),D_h=m.period(a,1,start,end),V_h=m.period(a,2,start,end),network_h=m.period(a,4,start,end)))
    table('natural_vs_network_windows.csv',rows)
    report=dict(regime_cases=len(regime),regime_class_counts={str(k):int(v) for k,v in regime.state.value_counts().items()},T22_representative_scale=float(representative_scale),scope='Finite last-14-day exposure window; numerical classification is not an asymptotic proof.')
    (VALIDATION/'additional_analyses.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

def landscape():
    # A local sensitivity slice, evaluated with the same final observable score.
    import fit_protocol as fit
    from common import chosen
    pv=np.asarray(chosen('D-period-X')['vector']);rows=[]
    matched_bounds=dict(zip(fit.model.FREE['D-period-X'],fit.bounds('D-period-X')))
    gp_bounds=matched_bounds['gp_s'];gphi_bounds=matched_bounds['gphi_s']
    gp_axis=np.linspace(max(gp_bounds[0],pv[6]-.35),min(gp_bounds[1],pv[6]+.35),25)
    gphi_axis=np.linspace(max(gphi_bounds[0],pv[5]-1.25),min(gphi_bounds[1],pv[5]+1.25),25)
    for gp in gp_axis:
        for gphi in gphi_axis:
            q=pv.copy();q[5]=gphi;q[6]=gp;iv,ev,_=fit.rescore(q,'D-period-X')
            rows.append(dict(gphi_d=gphi,gp_d=gp,weighted_sse=fit.final_score(iv,ev),dt_h=fit.DT_FINAL,objective='common_final_observable_penalty',selected_gphi_d=pv[5],selected_gp_d=pv[6]))
        print(f'Unified local objective slice {len(rows)}/625',flush=True)
    table('original_parameter_landscape.csv',rows)
    return dict(cases=len(rows),grid_points_per_axis=25,requested_halfwidth_gp=.35,requested_halfwidth_gphi=1.25,gp_range=[float(gp_axis.min()),float(gp_axis.max())],gphi_range=[float(gphi_axis.min()),float(gphi_axis.max())],clipped_to_shared_search_bounds=True,objective='common_final_observable_penalty',scope='Local two-coordinate slice with all remaining parameters fixed at selected fit; not a new search.')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--only',nargs='+',choices=list(GROUP_FILES))
    args=parser.parse_args(argv);VALIDATION.mkdir(exist_ok=True);stamp=VALIDATION/'unified_analyses.json'
    previous=json.loads(stamp.read_text()) if stamp.exists() else {};fp=fingerprint()
    report=previous if previous.get('fingerprint')==fp else dict(fingerprint=fp,parameters=dict(m.PARAMS),groups={})
    report['parameter_source']='reference_model.PARAMS (selected unified D-period-X fit)'
    for group in args.only or list(GROUP_FILES):
        old=report['groups'].get(group,{})
        fresh=bool(old) and all((DATA/n).is_file() and old.get('files',{}).get(n)==sha(DATA/n) for n in GROUP_FILES[group])
        if fresh and not args.force:print('Unified data current:',group,flush=True);continue
        start=time.perf_counter();print('Reintegrating unified analysis:',group,flush=True)
        info=globals()[group]()
        report['groups'][group]=dict(elapsed_seconds=time.perf_counter()-start,details=info,files={n:sha(DATA/n) for n in GROUP_FILES[group]})
        stamp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    report['complete']=all(g in report['groups'] for g in GROUP_FILES)
    stamp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':main()
