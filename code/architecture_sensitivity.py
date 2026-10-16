"""Full forward-model gain sensitivity around the matched-search winners.

Each point repeats acquisition and release; no frozen equilibrium or reconstructed
trajectory supplies any observed phase or period. Cached data are fingerprinted.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.optimize import brentq
from common import ROOT,DATA,chosen,ORDER
import architecture_model as model

DT=1/64
SCALES=np.linspace(0.,2.,81)
ARCHES=['D-period-X','coupling-X']
OUT=DATA/'Figure2_full_model_gain_sweep.csv'
META=ROOT/'validation/Figure2_gain_sweep_metadata.json'

def fingerprint():
    h=hashlib.sha256()
    for f in [Path(__file__),Path(model.__file__)]:h.update(f.read_bytes())
    for a in ARCHES:
        q=chosen(a);h.update(json.dumps({'architecture':a,'seed':q['seed'],'vector':q['vector']},sort_keys=True).encode())
    return h.hexdigest()

def simulate_point(arch,condition,scale,dt=DT):
    q=chosen(arch);p=np.asarray(q['vector']).copy()
    gain='g_d' if arch=='D-period-X' else 'g_coupling'
    idx=model.NAMES.index(gain);reference_gain=float(p[idx]);p[idx]*=scale
    summary,tr=model.simulate(arch,p,condition,ex=True,dt=dt)
    dd=tr[tr[:,0]>=0];early=dd[dd[:,0]<=168+1e-9];r=dd[0]
    phasor=np.mean(np.exp(1j*(early[:,5]-early[:,6])))
    dw=2*np.pi/r[12]-2*np.pi/r[13];ke=r[18]
    out=dict(architecture=arch,condition=condition,gain_parameter=gain,
             selected_seed=q['seed'],gain_scale=float(scale),reference_gain=reference_gain,
             gain_value=float(p[idx]),dt_h=dt,
             release_delta_omega_rad_h=float(dw),release_K_eff_rad_h=float(ke),
             release_memory_D=float(r[8]),release_memory_X=float(r[10]),
             release_gap_wrapped_h=float(r[15]),gap_circular_7d_h=float(np.angle(phasor)*24/(2*np.pi)),
             gap_resultant_7d=float(abs(phasor)),
             release_locking_ratio=float(abs(dw/ke)) if abs(ke)>1e-10 else np.inf,
             D_period_refined_h=model.refined_period(tr,1),V_period_refined_h=model.refined_period(tr,2),
             D_period_slope_h=summary['D_regression_h'],V_period_slope_h=summary['V_regression_h'],
             maximum_ex_DD_feedback_rad_h=float(np.max(abs(dd[:,16:18]))),
             maximum_preDD_feedback_rad_h=float(np.max(abs(tr[tr[:,0]<0,16:18]))))
    for name,col in [('D',1),('V',2)]:
        f,power=LombScargle(dd[:,0],dd[:,col],center_data=True,fit_mean=True).autopower(
            minimum_frequency=1/30,maximum_frequency=1/18,samples_per_peak=20)
        out[name+'_period_fit_grid_h']=float(1/f[np.argmax(power)])
    return out

def refine_coupling_crossings(z):
    """Resolve zero and nearby locking boundaries with new full simulations."""
    arch='coupling-X';q=chosen(arch);base=np.asarray(q['vector'])
    dw=abs(2*np.pi/base[0]-2*np.pi/base[1]);extra=[]
    for c in ORDER:
        sub=z[(z.architecture==arch)&(z.condition==c)].sort_values('gain_scale')
        scales=sub.gain_scale.to_numpy();ks=sub.release_K_eff_rad_h.to_numpy()
        for i in np.flatnonzero(ks[:-1]*ks[1:]<0):
            lo,hi=scales[i:i+2]
            def driver(s):
                p=base.copy();p[model.NAMES.index('g_coupling')]*=s
                _,tr=model.simulate(arch,p,c,ex=True,dt=DT)
                return tr[np.flatnonzero(tr[:,0]>=0)[0],18]
            targets=[0.]+[sign*fac*dw for sign in [-1,1] for fac in [1/64,1/16,1/4,1/2,1,2,4]]
            for target in targets:
                if (ks[i]-target)*(ks[i+1]-target)>0:continue
                scale=brentq(lambda s:driver(s)-target,lo,hi,xtol=1e-12)
                r=simulate_point(arch,c,scale);r['sample_design']='targeted near coupling zero';extra.append(r)
    if extra:z=pd.concat([z,pd.DataFrame(extra)],ignore_index=True)
    return z.sort_values(['architecture','condition','gain_scale']).reset_index(drop=True)

def validate_convergence(z):
    cases=set()
    for arch in ARCHES:
        for c in ['LD','T22','T26']:
            for s in [0.,1.,2.]:cases.add((arch,c,s))
        for c in ORDER:
            q=z[(z.architecture==arch)&(z.condition==c)].sort_values('gain_scale')
            # Add the closest sampled locking boundary and both ends of the
            # largest circular-phase step, where a fixed step size is most at risk.
            ids=[(q.release_locking_ratio-1).abs().idxmin()]
            phase=q.gap_circular_7d_h.to_numpy()
            jump=np.abs((np.diff(phase)+12)%24-12)
            j=int(np.argmax(jump));ids.extend([q.index[j],q.index[j+1]])
            for idx in ids:cases.add((arch,c,float(z.loc[idx,'gain_scale'])))
    rows=[]
    for arch,c,s in sorted(cases):
        old=z[(z.architecture==arch)&(z.condition==c)&np.isclose(z.gain_scale,s)].iloc[0]
        fine=simulate_point(arch,c,s,DT/2)
        phase=abs((old.gap_circular_7d_h-fine['gap_circular_7d_h']+12)%24-12)
        rows.append(dict(architecture=arch,condition=c,gain_scale=s,
                         coarse_dt_h=DT,fine_dt_h=DT/2,
                         D_period_difference_h=abs(old.D_period_refined_h-fine['D_period_refined_h']),
                         V_period_difference_h=abs(old.V_period_refined_h-fine['V_period_refined_h']),
                         circular_phase_difference_h=phase,
                         delta_omega_difference_rad_h=abs(old.release_delta_omega_rad_h-fine['release_delta_omega_rad_h']),
                         K_eff_difference_rad_h=abs(old.release_K_eff_rad_h-fine['release_K_eff_rad_h']),
                         coarse_resultant=old.gap_resultant_7d,fine_resultant=fine['gap_resultant_7d']))
    cv=pd.DataFrame(rows);cv.to_csv(ROOT/'tables/Figure2_gain_sweep_convergence.csv',index=False)
    return cv

def build(force=False):
    fp=fingerprint()
    if not force and OUT.exists() and META.exists():
        old=json.loads(META.read_text())
        if old.get('fingerprint')==fp:return pd.read_csv(OUT)
    rows=[]
    for arch in ARCHES:
        for i,s in enumerate(SCALES):
            for c in ORDER:rows.append(simulate_point(arch,c,float(s)))
            if i%20==0:print(f'{arch}: full-model gain {i+1}/{len(SCALES)}',flush=True)
    z=pd.DataFrame(rows);z['sample_design']='uniform gain grid'
    z=refine_coupling_crossings(z);z.to_csv(OUT,index=False)
    cv=validate_convergence(z)
    baseline=[]
    for a in ARCHES:
        fit=pd.DataFrame(chosen(a)['selected']['endpoints'])
        for c in ORDER:
            r=z[(z.architecture==a)&(z.condition==c)&np.isclose(z.gain_scale,1)].iloc[0]
            b=fit[(fit.condition==c)&fit.ex_vivo].iloc[0]
            baseline.append(max(abs(r.D_period_fit_grid_h-b.D_h),abs(r.V_period_fit_grid_h-b.V_h)))
    meta=dict(fingerprint=fp,rows=len(z),uniform_gain_scales=len(SCALES),
              targeted_crossing_samples=int(z.sample_design.ne('uniform gain grid').sum()),architectures=ARCHES,
              gain_parameters={'D-period-X':'g_d','coupling-X':'g_coupling'},
              selected_seeds={a:chosen(a)['seed'] for a in ARCHES},dt_h=DT,
              protocol='Complete 720 h exposure plus 336 h ex vivo DD at every gain. X feedback removed only at DD.',
              driver='Actual instantaneous full-model DD-release delta_omega or K_eff.',
              phase='Circular mean of phi_D-phi_V over saved DD0-168 h, wrapped to [-12,12) phase-equivalent h.',
              displayed_period='architecture_model.refined_period: DD0-336 h; Lomb-Scargle search over 12-40 h at 12 samples/peak, followed by bounded refinement. This differs from the 18-30 h discrete fitting grid.',
              fit_grid_period='Original discrete LS grid, 18-30 h, 20 samples/peak; retained for selected-fit equality.',
              ratio='Absolute release delta_omega/K_eff, a frozen diagnostic, not a measured locking classification.',
              ratio_poles='Values with absolute release K_eff <= 1e-10 rad/h are recorded as infinity and omitted from connecting curves. Full-model samples target both sides of each coupling-zero crossing.',
              convergence_cases=len(cv),
              maximum_checked_period_difference_h=float(cv[['D_period_difference_h','V_period_difference_h']].max().max()),
              maximum_checked_circular_phase_difference_h=float(cv.circular_phase_difference_h.max()),
              minimum_phase_resultant=float(z.gap_resultant_7d.min()),
              maximum_baseline_fit_grid_difference_h=float(max(baseline)),
              all_ex_DD_feedback_zero=bool(z.maximum_ex_DD_feedback_rad_h.eq(0).all()),
              all_preDD_feedback_retained=bool(z.maximum_preDD_feedback_rad_h.gt(0).all()),
              historical_audit='The previous original_logic_sensitivity table used an illustrative coupling-only gain fit and seven-day-averaged drivers. Its observations were not reused here.')
    META.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    assert meta['maximum_baseline_fit_grid_difference_h']<1e-10
    assert meta['all_ex_DD_feedback_zero'] and meta['all_preDD_feedback_retained']
    print(json.dumps(meta,indent=2),flush=True)
    return z

if __name__=='__main__':build()
