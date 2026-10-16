from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from astropy.timeseries import LombScargle
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
ROOT=Path(__file__).resolve().parents[1]
DT=.03125
ORDER=['SP','LP','LD','T22','T26']
SCHEDULES={'SP':(24,8),'LP':(24,16),'LD':(24,12),'T22':(22,11),'T26':(26,13),'18:6 LD':(24,18),'T23':(23,11.5),'T25':(25,12.5)}
from parameter_registry import reference_parameters
PARAMS=reference_parameters()
KEYS=['tau_d0','tau_v','tau_x0','g_d','g_x','gphi_d','gp_d','gphi_x','gp_x','kvd','kdv','kx_d','kx_v','ksx','phase_bias','light_amp','v_light_boost','mem_tau_d','mem_tau_x']
COL=['time_from_DD_h','D','V','X','behavior','phi_D','phi_V','phi_X','memory_D','memory_X','tau_D','tau_X','gap_full_h','gap_folded_h','X_to_D_rad_h','X_to_V_rad_h','K_eff','output_resultant','phi_output','drive_D','drive_X']
@njit(cache=True)
def integrate(p,cycle,photo,dt,release,ddh,initial,wr,wf,xret,cscale,
              mret,kick_h,cscale_before,xfeed_before,gate):
    n=int(round((release+ddh)/dt))+1
    out=np.zeros((n,21)); d,v,x=initial[0],initial[1],initial[2]
    md,mx=0.,0.; twopi=2*np.pi; changed=False
    for i in range(n):
        t=i*dt; isdd=t>=release-1e-9
        if isdd and not changed:
            # D alone is advanced by kick_h circadian hours; V and X unchanged.
            d=(d+kick_h*twopi/24)%twopi
            md*=mret; mx*=mret; changed=True
        light=0. if isdd else (1. if t%cycle<photo else 0.)
        theta=0. if isdd else twopi*(t%cycle)/cycle
        sx=(1+np.sin(x))/2; sd=(1+np.sin(d))/2
        drd=p[5]*light*np.sin(theta-d)+p[6]*(sd-light)
        drx=p[7]*light*np.sin(theta-x)+p[8]*(sx-light)
        if isdd and gate: drd,drx=0.,0.
        md+=dt*(drd-md)/p[17]; mx+=dt*(drx-mx)/p[18]
        td=min(34.,max(16.,p[0]-p[3]*np.tanh(md)))
        tx=min(34.,max(16.,p[2]-p[4]*np.tanh(mx)))
        scale=cscale if isdd else cscale_before
        kvd=p[9]*scale; kdv=p[10]*scale
        sr=wr*np.sin(d)+(1-wr)*np.sin(v); cr=wr*np.cos(d)+(1-wr)*np.cos(v)
        read=np.arctan2(sr,cr)
        feed=np.arctan2(wf*np.sin(d)+(1-wf)*np.sin(v),wf*np.cos(d)+(1-wf)*np.cos(v))
        factor=xret if isdd else xfeed_before
        fxd=factor*p[11]*np.sin(x-d); fxv=factor*p[12]*np.sin(x-v)
        gap=((d-v+np.pi)%twopi-np.pi)*24/twopi
        folded=gap-12 if gap>6 else (gap+12 if gap< -6 else gap)
        out[i]=np.array([t-release,-np.sin(d),-np.sin(v),-np.sin(x),-np.sin(read),d,v,x,md,mx,td,tx,gap,folded,fxd,fxv,kvd-kdv,np.sqrt(sr*sr+cr*cr),read,drd,drx])
        dd=twopi/td-kdv*np.sin(v-d+p[14])+fxd+p[15]*light*np.sin(theta-d)
        dv=twopi/p[1]+kvd*np.sin(d-v-p[14])+fxv+p[16]*p[15]*light*np.sin(theta-v)
        dx=twopi/tx+p[13]*np.sin(feed-x)
        d=(d+dt*dd)%twopi; v=(v+dt*dv)%twopi; x=(x+dt*dx)%twopi
    return out

def simulate(c,ex=False,dt=DT,release=720.,ddh=336.,initial=(.35,0.,-.2),
             wr=.8,wf=.8,xret=None,cscale=1.,mret=1.,kick_h=0.,
             cscale_before=1.,xfeed_before=1.,gate=True,changes=None):
    p=dict(PARAMS); p.update(changes or {})
    return integrate(np.array([p[k] for k in KEYS]),*SCHEDULES[c],dt,release,
      ddh,np.array(initial),wr,wf,float(not ex) if xret is None else xret,
      cscale,mret,kick_h,cscale_before,xfeed_before,gate)

def period(a,k,start=0.,end=336.,refined=True,samples=20):
    s=a[(a[:,0]>=start-1e-8)&(a[:,0]<=end+1e-8)]
    stride=max(1,int(round(.25/(s[1,0]-s[0,0])))); s=s[::stride]
    ls=LombScargle(s[:,0],s[:,k],center_data=True,fit_mean=True)
    f,p=ls.autopower(minimum_frequency=1/30,maximum_frequency=1/18,samples_per_peak=samples)
    j=int(np.argmax(p))
    if not refined or j==0 or j==len(f)-1: return float(1/f[j])
    opt=minimize_scalar(lambda q:-float(ls.power(q,method='cython')),bounds=(f[j-1],f[j+1]),method='bounded',options={'xatol':1e-12})
    return float(1/opt.x)

def first_peak(a,k):
    s=a[(a[:,0]>=0)&(a[:,0]<=36)]
    ix=find_peaks(s[:,k])[0]
    if not len(ix): return np.nan
    i=ix[0]; y=s[i-1:i+2,k]; offset=.5*(y[0]-y[2])/(y[0]-2*y[1]+y[2])
    return float(s[i,0]+offset*(s[1,0]-s[0,0]))

def metrics(a):
    dd=a[a[:,0]>=-1e-9]; first=dd[dd[:,0]<=168+1e-8]
    circ=np.angle(np.mean(np.exp(1j*first[:,12]*2*np.pi/24)))*24/(2*np.pi)
    pg=(first_peak(a,2)-first_peak(a,1)+12)%24-12
    return {'behavior_h':period(a,4),'D_h':period(a,1),'V_h':period(a,2),
            'gap_day1_mean_h':float(np.mean(dd[dd[:,0]<=24,12])),
            'gap_days1_4_mean_h':float(np.mean(dd[dd[:,0]<=96,12])),
            'gap_days1_7_mean_h':float(np.mean(first[:,12])),
            'gap_days1_7_circular_h':float(circ),'gap_first_peak_h':pg,
            'gap_release_h':float(dd[0,12]),'gap_legacy_folded_h':float(np.mean(first[:,13])),
            'fraction_folded_first7d':float(np.mean(np.abs(first[:,12]-first[:,13])>1)),
            'tau_D_mean_first7d_h':float(np.mean(first[:,10])),'tau_X_mean_first7d_h':float(np.mean(first[:,11])),
            'mean_X_to_D_first7d':float(np.mean(first[:,14])),'min_output_resultant':float(np.min(dd[:,17]))}

def locking(a,start,end,cycle):
    s=a[(a[:,0]>=start-1e-8)&(a[:,0]<=end+1e-8)]
    t=s[:,0]; d=np.unwrap(s[:,5]); v=np.unwrap(s[:,6]); delta=d-v
    sd,sv=np.polyfit(t,d,1)[0],np.polyfit(t,v,1)[0]
    pd_,pv=2*np.pi/sd,2*np.pi/sv
    # Continuous drift and resultant are primary; threshold is a descriptive
    # finite-window diagnostic, never proof that a system is locked/asymptotic.
    drift=(delta[-1]-delta[0])/(2*np.pi)
    range_cycles=np.ptp(delta)/(2*np.pi)
    tlight=2*np.pi*(t+720)/cycle
    return dict(D_slope_period_h=pd_,V_slope_period_h=pv,
      D_minus_V_slope_period_h=pd_-pv,
      relative_phase_net_drift_cycles=drift,relative_phase_range_cycles=range_cycles,
      relative_phase_drift_h_per_day=(sd-sv)*24*24/(2*np.pi),
      DV_locking_resultant=float(abs(np.mean(np.exp(1j*delta)))),
      D_light_locking_resultant=float(abs(np.mean(np.exp(1j*(d-tlight))))),
      V_light_locking_resultant=float(abs(np.mean(np.exp(1j*(v-tlight))))),
      net_relative_phase_slips=int(np.floor((delta[-1]+np.pi)/(2*np.pi))-np.floor((delta[0]+np.pi)/(2*np.pi))),
      resolved_frequency_split=bool(abs(pd_-pv)>.1 and abs(drift)>.25),
      D_entrained_frequency=bool(abs(pd_-cycle)<.1),V_entrained_frequency=bool(abs(pv-cycle)<.1))
