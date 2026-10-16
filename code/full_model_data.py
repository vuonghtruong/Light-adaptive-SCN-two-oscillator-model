"""Regenerate reference/prediction trajectories by integrating full D,V,X,MD,MX.

No reduced-model state or reconstructed phase enters these data.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import reference_model as model
import parameter_registry

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=model.ORDER+['18:6 LD','T23','T25']
DT=.015625

def fingerprint():
    payload=json.dumps(model.PARAMS,sort_keys=True).encode()
    for name in ['full_model_data.py','reference_model.py']:
        payload+=(ROOT/'code'/name).read_bytes()
    return hashlib.sha256(payload).hexdigest()

def main(force=False):
    report_path=ROOT/'validation/full_model_dynamic_data.json'
    trace_path=ROOT/'data/full_model_dynamic_traces.csv.gz'
    fp=fingerprint()
    if not force and report_path.exists() and trace_path.exists():
        cached=json.loads(report_path.read_text())
        if cached.get('fingerprint')==fp and cached.get('trace_sha256')==hashlib.sha256(trace_path.read_bytes()).hexdigest():
            pd.DataFrame(cached['endpoints']).to_csv(ROOT/'tables/full_model_dynamic_endpoints.csv',index=False)
            return cached
    traces=[];endpoints=[];checks=[]
    for condition in CONDITIONS:
        pair=[]
        for ex in [False,True]:
            a=model.simulate(condition,ex=ex,dt=DT)
            r=model.metrics(a)
            r.update(condition=condition,ex_vivo=ex,X_h=model.period(a,3))
            release=a[np.flatnonzero(a[:,0]>=0)[0]]
            r['delta_omega_release_rad_h']=2*np.pi/release[10]-2*np.pi/model.PARAMS['tau_v']
            r['K_eff_release_rad_h']=release[16]
            r['locking_ratio_release']=abs(r['delta_omega_release_rad_h']/r['K_eff_release_rad_h'])
            endpoints.append(r)
            stored=a[a[:,0]>=-336][::int(round(.25/DT))]
            table=pd.DataFrame(stored,columns=model.COL)
            table['condition']=condition;table['context']='exvivo' if ex else 'invivo'
            table['model']='full_original_D_V_X';traces.append(table);pair.append(a)
        inv,exv=pair;before=inv[:,0]<0;after=inv[:,0]>=0
        pre=float(np.max(abs(inv[before]-exv[before])))
        feedback=float(np.max(abs(exv[after,14:16])))
        release_difference=float(np.max(abs(inv[after][0,5:10]-exv[after][0,5:10])))
        checks.append(dict(condition=condition,pre_DD_all_state_difference=pre,exvivo_DD_feedback_max=feedback,release_phase_memory_difference=release_difference,passed=pre==0 and feedback==0 and release_difference<1e-12))
        print('Full-model trajectories:',condition,flush=True)
    out=pd.concat(traces,ignore_index=True)
    out.to_csv(trace_path,index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(endpoints).to_csv(ROOT/'tables/full_model_dynamic_endpoints.csv',index=False)
    # Step halving covers both context branches of all three novel schedules.
    refinement=[]
    for c in ['18:6 LD','T23','T25']:
        for ex in [False,True]:
            a=model.simulate(c,ex=ex,dt=DT/2)
            fine=model.metrics(a);r=next(r for r in endpoints if r['condition']==c and r['ex_vivo']==ex)
            period_error=max(abs(fine[k]-r[k]) for k in ['behavior_h','D_h','V_h'])
            phase_error=abs((fine['gap_days1_7_circular_h']-r['gap_days1_7_circular_h']+12)%24-12)
            refinement.append(dict(condition=c,ex_vivo=ex,maximum_period_difference_h=period_error,circular_gap_difference_h=phase_error,passed=period_error<.002 and phase_error<.02))
    report={'fingerprint':fp,'full_original_model':True,'integrated_states':['phi_D','phi_V','phi_X','memory_D','memory_X'],'reduced_phase_used':False,'parameter_source':'best penalized D-period-X','selected_seed':parameter_registry.selected()['seed'],'parameter_fingerprint':parameter_registry.fingerprint(),'parameters':model.PARAMS,'integration_step_h':DT,'stored_step_h':.25,'exposure_h':720,'DD_h':336,'conditions':CONDITIONS,'branch_checks':checks,'step_halving':refinement,'endpoints':endpoints,'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),'passed':all(c['passed'] for c in checks+refinement)}
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise AssertionError(report)
    print('Full dynamics validated; stored rows:',len(out),flush=True)
    return report

if __name__=='__main__':main()
