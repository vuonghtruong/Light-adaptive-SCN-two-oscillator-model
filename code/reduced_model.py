from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from numba import njit
from astropy.timeseries import LombScargle
ROOT=Path(__file__).resolve().parents[1]
import architecture_model as original
import fit_protocol as fit
OUT=ROOT/'work/reduction';OUT.mkdir(exist_ok=True,parents=True)
COLS=['time_from_DD_h','D','V','X','behavior','phi_D','phi_V','phi_X','memory_D','memory_V','memory_X','H','tau_D','tau_V','tau_X','gap_full_h','feedback_to_D_rad_h','feedback_to_V_rad_h','K_eff','output_resultant','phi_output','drive_D','drive_V','drive_X_or_H','lag','valid','contraction','light_on']

@njit(cache=True)
def root(p,mode,d,v,md,mx,light,theta,factor,coherent):
    delta=d-v; a=.8;b=.2;r=np.sqrt(a*a+b*b+2*a*b*np.cos(delta))
    psi=np.arctan2(a*np.sin(d)+b*np.sin(v),a*np.cos(d)+b*np.cos(v))
    wd=a*(a+b*np.cos(delta))/(r*r);wv=1-wd
    td=min(34.,max(16.,p[0]-p[3]*np.tanh(md))) if mode==0 else p[0]
    tv=p[1];tx=min(34.,max(16.,p[2]-p[4]*np.tanh(mx)))
    kvd=p[9]+(p[17]*np.tanh(md) if mode==1 else 0.)
    f0d=2*np.pi/td+p[10]*np.sin(delta)+p[14]*light*np.sin(theta-d)
    f0v=2*np.pi/tv+kvd*np.sin(delta)+p[15]*p[14]*light*np.sin(theta-v)
    if coherent:
        gg=factor*(wd*p[11]+wv*p[12])
        jj=factor*(-wd*p[11]*b+wv*p[12]*a)*np.sin(delta)
    else:
        gg=factor*(wd*p[11]*(a+b*np.cos(delta))+wv*p[12]*(b+a*np.cos(delta)))/r
        jj=factor*(-wd*p[11]*b+wv*p[12]*a)*np.sin(delta)/r
    cc=p[13]+gg;strength=np.hypot(cc,jj)
    mismatch=2*np.pi/tx-wd*f0d-wv*f0v
    valid=abs(mismatch)<strength
    alpha=np.arcsin(min(1.,max(-1.,mismatch/strength)))-np.arctan2(jj,cc)
    if coherent:
        fd=factor*p[11]*(np.sin(alpha)-b*np.cos(alpha)*np.sin(delta))
        fv=factor*p[12]*(np.sin(alpha)+a*np.cos(alpha)*np.sin(delta))
    else:
        fd=factor*p[11]*(a*np.sin(alpha)+b*np.sin(alpha-delta))/r
        fv=factor*p[12]*(a*np.sin(alpha+delta)+b*np.sin(alpha))/r
    return psi,alpha,fd,fv,f0d,f0v,td,tv,tx,kvd,r,valid,np.sqrt(max(0.,strength*strength-mismatch*mismatch))

@njit(cache=True)
def run(p,mode,cycle,photo,dt,ex,coherent):
    d=.35;v=0.;md=0.;mv=0.;mx=0.
    stride=int(round(.25/dt));n=int(round(1056/dt))+1
    out=np.empty(((n+stride-1)//stride,28));j=0;invalid_ld=0;invalid_dd=0
    for i in range(n):
        t=i*dt;rel=t-720;dd=t>=720-1e-9
        light=0. if dd else (1. if t%cycle<photo else 0.)
        theta=0. if dd else 2*np.pi*(t%cycle)/cycle
        factor=0. if ex and dd else 1.
        z=root(p,mode,d,v,md,mx,light,theta,factor,coherent)
        x=z[0]+z[1] # Algebraic readout only: no integrated X phase.
        ud=p[5]*light*np.sin(theta-d)+p[6]*((1+np.sin(d))/2-light)
        uv=p[5]*light*np.sin(theta-v)+p[6]*((1+np.sin(v))/2-light)
        ux=p[7]*light*np.sin(theta-x)+p[8]*((1+np.sin(x))/2-light)
        if dd:ud=0.;uv=0.;ux=0.
        md+=dt*(ud-md)/240.;mv+=dt*(uv-mv)/240.;mx+=dt*(ux-mx)/240.
        psi,alpha,fd,fv,f0d,f0v,td,tv,tx,kvd,r,valid,lam=root(p,mode,d,v,md,mx,light,theta,factor,coherent)
        x=psi+alpha
        if not valid:
            if dd:invalid_dd+=1
            else:invalid_ld+=1
        if i%stride==0:
            gap=((d-v+np.pi)%(2*np.pi)-np.pi)*24/(2*np.pi)
            out[j]=np.array([rel,-np.sin(d),-np.sin(v),-np.sin(x),-np.sin(psi),d,v,x,md,mv,mx,np.nan,td,tv,tx,gap,fd,fv,kvd-p[10],r,psi,ud,uv,ux,alpha,(1. if valid else 0.),lam,light]);j+=1
        d=(d+dt*(f0d+fd))%(2*np.pi);v=(v+dt*(f0v+fv))%(2*np.pi)
    return out[:j],invalid_ld,invalid_dd

def metrics(tr):
    dd=tr[(tr[:,0]>=0)&(tr[:,0]<=336)]
    s=np.zeros(10)
    refined=[]
    for j,col in enumerate((4,1,2,3)):
        freq,power=LombScargle(dd[:,0],dd[:,col],center_data=True,fit_mean=True).autopower(minimum_frequency=1/30,maximum_frequency=1/18,samples_per_peak=20)
        s[j]=1/freq[np.argmax(power)]
        if j<3:refined.append(original.refined_period(tr,col))
    gap=dd[:,15].copy();gap[gap>6]-=12;gap[gap< -6]+=12
    s[4]=gap[0];s[5]=gap[dd[:,0]<=168].mean()
    s[6]=dd[dd[:,0]<=168,12].mean();s[7]=dd[dd[:,0]<=168,14].mean()
    ld=tr[(tr[:,0]>=-168)&(tr[:,0]<0)];s[8]=ld[:,8].mean();s[9]=ld[:,10].mean()
    return s,refined

def main():
    selected=[p for p in json.loads((ROOT/'data/reduction_parameters.json').read_text()) if p['architecture'] in ('D-period-X','coupling-X') and p['objective'] in ('submitted_reference','main_protocol')]
    rows=[];scores=[];traces=[]
    for p in selected:
        pv=np.array(p['vector']);mode=original.ARCHITECTURES[p['architecture']]
        for variant in ('oscillatory_X','adiabatic_exact_geometry','adiabatic_coherent'):
            inv=np.empty((5,10));ex=np.empty((5,10));bad=0
            for i,c in enumerate(original.ORDER):
                for isex,arr in [(False,inv),(True,ex)]:
                    if variant=='oscillatory_X':
                        _,tr=original.simulate(p['architecture'],pv,c,ex=isex,dt=.015625)
                        extra=np.full((len(tr),4),np.nan)
                        extra[:,0]=original.wrap(tr[:,7]-tr[:,20]);extra[:,1]=1
                        t=tr[:,0]+720;extra[:,3]=((tr[:,0]<0)&((t%original.SCHEDULES[c][0])<original.SCHEDULES[c][1])).astype(float)
                        tr=np.column_stack([tr,extra]);nld=ndd=0
                    else:tr,nld,ndd=run(pv,mode,*original.SCHEDULES[c],.015625,isex,variant=='adiabatic_coherent')
                    m,refined=metrics(tr);arr[i]=m;bad+=nld+ndd
                    row=dict(fit_id=p['fit_id'],architecture=p['architecture'],objective=p['objective'],variant=variant,condition=c,context='exvivo' if isex else 'invivo',invalid_LD_steps=nld,invalid_DD_steps=ndd)
                    for name,value in zip(['behavior_h','D_h','V_h','X_h','gap_release_h','gap_7day_h'],m[:6]):row[name]=value
                    for name,value in zip(['behavior_refined_h','D_refined_h','V_refined_h'],refined):row[name]=value
                    rows.append(row)
                    tab=pd.DataFrame(tr,columns=COLS)
                    for key in ('fit_id','architecture','objective','variant','condition','context'):tab[key]=row[key]
                    traces.append(tab)
            scores.append(dict(fit_id=p['fit_id'],variant=variant,invalid_steps=bad,shared_final_score=fit.final_score(inv,ex),shared_screening_score=float(fit.screening_score(inv,ex,False))))
            print(p['fit_id'],variant,'invalid',bad,'score',scores[-1]['shared_final_score'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'target_comparison.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'score_comparison.csv',index=False)
    pd.concat(traces,ignore_index=True).to_csv(OUT/'independent_traces.csv.gz',index=False)
    (OUT/'parameters.json').write_text(json.dumps(selected,indent=2))

if __name__=='__main__':main()

