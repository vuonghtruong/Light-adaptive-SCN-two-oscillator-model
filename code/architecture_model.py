"""Explicit matched model alternatives. No literature values are invented here.

The D-period-X branch reproduces the archived forward-Euler implementation;
all branches use the same update order, DD gate, and 30 d + 14 d protocol.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT/'cache'/'numba_architecture'))
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar
from astropy.timeseries import LombScargle

ORDER = ['SP', 'LP', 'LD', 'T22', 'T26']
SCHEDULES = {'SP':(24.,8.),'LP':(24.,16.),'LD':(24.,12.),'T22':(22.,11.),'T26':(26.,13.),'18:6 LD':(24.,18.),'T23':(23.,11.5),'T25':(25.,12.5)}
ARCHITECTURES = {'D-period-X':0, 'coupling-X':1, 'V-period-X':2, 'joint-DV-period-X':3, 'D-period-H':4, 'Y-X':5}
# Array positions shared by every implementation; inactive entries are fixed.
NAMES = ['tau_d0','tau_v0','tau_x0','g_d','g_x','gphi_s','gp_s','gphi_x','gp_x','kvd','kdv','kx_d','kx_v','ksx','light_amp','v_light_boost','g_v','g_coupling','gphi_h','gp_h','kh_d','kh_v']
BOUNDS = {'tau_d0':(21.5,24.5),'tau_v0':(23.,25.8),'tau_x0':(20.5,28.),'g_d':(0.,40.),'g_x':(0.,40.),'gphi_s':(-5.,5.),'gp_s':(-6.,6.),'gphi_x':(-5.,5.),'gp_x':(-6.,6.),'kvd':(.01,.075),'kdv':(.002,.05),'kx_d':(0.,.5),'kx_v':(0.,.3),'ksx':(0.,.5),'light_amp':(.005,.5),'v_light_boost':(1.,60.),'g_v':(0.,40.),'g_coupling':(-.15,.15),'gphi_h':(-5.,5.),'gp_h':(-6.,6.),'kh_d':(0.,.5),'kh_v':(0.,.3)}
COMMON = ['tau_d0','tau_v0','tau_x0','g_x','gphi_s','gp_s','gphi_x','gp_x','kvd','kdv','kx_d','kx_v','ksx','light_amp','v_light_boost']
FREE = {'D-period-X':COMMON+['g_d'], 'coupling-X':COMMON+['g_coupling'], 'V-period-X':COMMON+['g_v'], 'joint-DV-period-X':COMMON+['g_d','g_v'], 'D-period-H':['tau_d0','tau_v0','g_d','gphi_s','gp_s','kvd','kdv','light_amp','v_light_boost','gphi_h','gp_h','kh_d','kh_v'], 'Y-X':['tau_d0','tau_x0','g_d','g_x','gphi_s','gp_s','gphi_x','gp_x','kx_d','ksx','light_amp']}
TRACE_COLUMNS = ['time_from_DD_h','D','V','X','behavior','phi_D','phi_V','phi_X','memory_D','memory_V','memory_X','H','tau_D','tau_V','tau_X','gap_full_h','feedback_to_D_rad_h','feedback_to_V_rad_h','K_eff','output_resultant','phi_output','drive_D','drive_V','drive_X_or_H']
SUMMARY_COLUMNS = ['behavior_regression_h','D_regression_h','V_regression_h','gap_first_peak_h','gap_days1_7_circular_h','tau_D_mean_first7d_h','tau_V_mean_first7d_h','tau_X_mean_first7d_h','H_mean_first7d','minimum_output_resultant']

def vector(architecture, normalized):
    p = np.zeros(len(NAMES)); p[:3] = 24.
    for key,z in zip(FREE[architecture],normalized):
        lo,hi=BOUNDS[key]; p[NAMES.index(key)] = lo+(hi-lo)*float(z)
    return p

@njit(cache=True)
def wrap(z):
    return (z+np.pi)%(2*np.pi)-np.pi

@njit(cache=True)
def run(p, mode, cycle, photo, dt=.125, ex=False, save=False, initial_d=.35, initial_v=0., initial_x=-.2):
    twopi=2*np.pi; release=720.; ddh=336.
    n=int(round((release+ddh)/dt))+1
    stride=max(1,int(round(.25/dt)))
    trace=np.empty(((n+stride-1)//stride,24)) if save else np.empty((0,24))
    d,v,x=initial_d,initial_v,initial_x
    md,mv,mx,h=0.,0.,0.,0.
    sums=np.zeros((3,5)); offsets=np.zeros(3); prev=np.zeros(3)
    peak=np.full(2,np.nan); yprev2=np.zeros(2); yprev=np.zeros(2)
    sum_sin,sum_cos=0.,0.; sum_td,sum_tv,sum_tx,sum_h=0.,0.,0.,0.; ng=0
    minresultant=1.; havedd=False; nt=0; ndd=0
    for i in range(n):
        t=i*dt; rel=t-release; isdd=t>=release-1e-9
        light=0. if isdd else (1. if t%cycle<photo else 0.)
        theta=0. if isdd else twopi*(t%cycle)/cycle
        sy=.8*np.sin(d)+.2*np.sin(v); cy=.8*np.cos(d)+.2*np.cos(v)
        read=np.arctan2(sy,cy) if mode!=5 else d
        drd=p[5]*light*np.sin(theta-d)+p[6]*((1+np.sin(d))/2-light)
        drv=p[5]*light*np.sin(theta-v)+p[6]*((1+np.sin(v))/2-light)
        drx=p[7]*light*np.sin(theta-x)+p[8]*((1+np.sin(x))/2-light)
        drh=p[18]*light*np.sin(theta-read)+p[19]*((1+np.sin(read))/2-light)
        if isdd: drd,drv,drx,drh=0.,0.,0.,0.
        md+=dt*(drd-md)/240.; mv+=dt*(drv-mv)/240.; mx+=dt*(drx-mx)/240.; h+=dt*(drh-h)/240.
        # Coupling memory follows D; V-only adaptation has V's own phase drive.
        td=p[0]-p[3]*np.tanh(md) if mode in (0,3,4,5) else p[0]
        tv=p[1]-p[16]*np.tanh(mv) if mode in (2,3) else p[1]
        tx=p[2]-p[4]*np.tanh(mx)
        td=min(34.,max(16.,td)); tv=min(34.,max(16.,tv)); tx=min(34.,max(16.,tx))
        kvd=p[9]+(p[17]*np.tanh(md) if mode==1 else 0.); kdv=p[10]
        factor=0. if ex and isdd else 1.
        fd=factor*(p[20]*np.tanh(h) if mode==4 else p[11]*np.sin(x-d))
        fv=factor*(p[21]*np.tanh(h) if mode==4 else p[12]*np.sin(x-v))
        gap=wrap(d-v)*24/twopi
        resultant=np.sqrt(sy*sy+cy*cy) if mode!=5 else 1.
        if save and i%stride==0:
            trace[nt]=np.array([rel,-np.sin(d),-np.sin(v),-np.sin(x),-np.sin(read),d,v,x,md,mv,mx,h,td,tv,tx,gap,fd,fv,kvd-kdv,resultant,read,drd,drv,drh if mode==4 else drx]); nt+=1
        if isdd:
            phases=np.array([read,d,v])
            for k in range(3):
                if havedd:
                    jump=phases[k]-prev[k]
                    if jump>np.pi: offsets[k]-=twopi
                    elif jump< -np.pi: offsets[k]+=twopi
                y=phases[k]+offsets[k]
                sums[k,0]+=1.; sums[k,1]+=rel; sums[k,2]+=y; sums[k,3]+=rel*rel; sums[k,4]+=rel*y
                prev[k]=phases[k]
            havedd=True; minresultant=min(minresultant,resultant)
            vals=np.array([-np.sin(d),-np.sin(v)])
            if ndd>=2 and rel<=36.+dt:
                for k in range(2):
                    if np.isnan(peak[k]) and yprev[k]>yprev2[k] and yprev[k]>=vals[k]:
                        denom=yprev2[k]-2*yprev[k]+vals[k]
                        frac=.5*(yprev2[k]-vals[k])/denom if denom!=0 else 0.
                        peak[k]=rel-dt+frac*dt
            yprev2=yprev.copy(); yprev=vals; ndd+=1
            if rel<=168.+1e-9:
                sum_sin+=np.sin(d-v); sum_cos+=np.cos(d-v)
                sum_td+=td;sum_tv+=tv;sum_tx+=tx;sum_h+=h;ng+=1
        dd=twopi/td-kdv*np.sin(v-d)+fd+p[14]*light*np.sin(theta-d)
        dv=twopi/tv+kvd*np.sin(d-v)+fv+p[15]*p[14]*light*np.sin(theta-v)
        dx=twopi/tx+p[13]*np.sin(read-x)
        if mode==5: dd=twopi/td+fd+p[14]*light*np.sin(theta-d); dv=0.
        if mode==4: dx=0.
        d=(d+dt*dd)%twopi;v=(v+dt*dv)%twopi;x=(x+dt*dx)%twopi
    periods=np.empty(3)
    for k in range(3):
        ns,sx,sy2,sxx,sxy=sums[k]
        slope=(ns*sxy-sx*sy2)/(ns*sxx-sx*sx)
        periods[k]=twopi/slope if slope>0 else np.nan
    pg=(peak[1]-peak[0]+12)%24-12
    summary=np.array([periods[0],periods[1],periods[2],pg,np.arctan2(sum_sin,sum_cos)*24/twopi,sum_td/ng,sum_tv/ng,sum_tx/ng,sum_h/ng,minresultant])
    if mode==5:
        summary[2]=np.nan; summary[3]=np.nan;summary[4]=np.nan;summary[6]=np.nan
    if mode==4: summary[7]=np.nan
    return summary,trace[:nt]

def simulate(architecture,p,condition,ex=False,dt=.015625,save=True):
    summary,trace=run(np.asarray(p),ARCHITECTURES[architecture],*SCHEDULES[condition],dt,ex,save)
    if save:
        if architecture=='Y-X': trace[:,[2,6,9,13,15,17,18,22]]=np.nan
        if architecture=='D-period-H':trace[:,[3,7,10,14]]=np.nan
        else:trace[:,11]=np.nan
    return dict(zip(SUMMARY_COLUMNS,summary)),trace

def refined_period(trace,column):
    s=trace[(trace[:,0]>=-1e-9)&(trace[:,0]<=336.+1e-9)]
    if not np.all(np.isfinite(s[:,column])): return np.nan
    ls=LombScargle(s[:,0],s[:,column],center_data=True,fit_mean=True)
    f,power=ls.autopower(minimum_frequency=1/40,maximum_frequency=1/12,samples_per_peak=12)
    j=int(np.argmax(power))
    if j==0 or j==len(f)-1:return float(1/f[j])
    opt=minimize_scalar(lambda z:-float(ls.power(z,method='cython')),bounds=(f[j-1],f[j+1]),method='bounded',options={'xatol':1e-12})
    return float(1/opt.x)

@njit(cache=True)
def evaluate(p,mode,dt,objective,behavior_targets,phase_targets):
    cycles=np.array([24.,24.,24.,22.,26.]); photos=np.array([8.,16.,12.,11.,13.])
    total=0.; endpoints=np.empty(10)
    for i in range(5):
        inv,_=run(p,mode,cycles[i],photos[i],dt,False,False)
        residual=inv[0]-behavior_targets[i]
        if not np.isfinite(residual):return 1e8,endpoints
        total+=residual*residual;endpoints[i]=inv[0]
        if objective!=2:
            ex,_=run(p,mode,cycles[i],photos[i],dt,True,False)
            phase=ex[3] if objective==0 else ex[4]
            residual=(phase-phase_targets[i]+12)%24-12
            if not np.isfinite(residual):return 1e8,endpoints
            total+=residual*residual;endpoints[5+i]=phase
        else:endpoints[5+i]=np.nan
    return total,endpoints
