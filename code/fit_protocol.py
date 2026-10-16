"""Archived two-stage search protocol with shared observable constraints.

Numerical branches and vector indexing are retained for reproducibility;
the package's comparisons use the four SCN architectures with oscillatory X.
"""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT; REV=ROOT
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from numba import njit,prange,set_num_threads
from astropy.timeseries import LombScargle
import architecture_model as model

(ROOT/'tables').mkdir(parents=True,exist_ok=True)
ORDER=model.ORDER; CONDITIONS=np.array([model.SCHEDULES[c] for c in ORDER])
TARGETS=pd.read_csv(PARENT/'data/archived_targets.csv').set_index('condition').loc[ORDER]
BT=TARGETS.behavior_mean_h.to_numpy(); ET=TARGETS.exvivo_scn_target_h.to_numpy()
GT=TARGETS.gap_mean_h.to_numpy(); GS=TARGETS.gap_sem_h.to_numpy()
DT=.0625; DT_FINAL=.015625; SEED=1729
LATIN=65536; LOCAL_STARTS=32; LOCAL_ITERS=55; FINALISTS=200
JOBS=[(a,'main_protocol') for a in model.ARCHITECTURES if a!='Y-X']+[
      ('D-period-X','behavior_only_main'),('Y-X','behavior_only_main')]


@njit(cache=True)
def summary(p,mode,cycle,photo,dt,ex):
    """Allocation-light scalar forward model; checked against architecture_model."""
    twopi=2*np.pi; d=.35; v=0.; x=-.2; md=mv=mx=h=0.
    # behavior,D,V,X phase regression statistics
    prev=np.zeros(4); offsets=np.zeros(4); sums=np.zeros((4,5)); seen=False
    gap_sum=td_sum=tx_sum=0.; ng=0; release_gap=0.; release_seen=False
    sd_sum=sx_sum=0.; ns=0
    for k in range(int(round(1056/dt))+1):
        t=k*dt; rel=t-720; dd=t>=720-1e-9
        light=0. if dd else (1. if t%cycle<photo else 0.)
        theta=0. if dd else twopi*(t%cycle)/cycle
        read=np.arctan2(.8*np.sin(d)+.2*np.sin(v),.8*np.cos(d)+.2*np.cos(v)) if mode!=5 else d
        ud=p[5]*light*np.sin(theta-d)+p[6]*((1+np.sin(d))/2-light)
        uv=p[5]*light*np.sin(theta-v)+p[6]*((1+np.sin(v))/2-light)
        ux=p[7]*light*np.sin(theta-x)+p[8]*((1+np.sin(x))/2-light)
        uh=p[18]*light*np.sin(theta-read)+p[19]*((1+np.sin(read))/2-light)
        if dd: ud=uv=ux=uh=0.
        md+=dt*(ud-md)/240.; mv+=dt*(uv-mv)/240.; mx+=dt*(ux-mx)/240.; h+=dt*(uh-h)/240.
        td=min(34.,max(16.,p[0]-p[3]*np.tanh(md))) if mode in (0,3,4,5) else p[0]
        tv=min(34.,max(16.,p[1]-p[16]*np.tanh(mv))) if mode in (2,3) else p[1]
        tx=min(34.,max(16.,p[2]-p[4]*np.tanh(mx)))
        kvd=p[9]+(p[17]*np.tanh(md) if mode==1 else 0.)
        factor=0. if ex and dd else 1.
        fd=factor*(p[20]*np.tanh(h) if mode==4 else p[11]*np.sin(x-d))
        fv=factor*(p[21]*np.tanh(h) if mode==4 else p[12]*np.sin(x-v))
        gap=((d-v+np.pi)%twopi-np.pi)*24/twopi
        # Exactly the additional +/-6 h fold used by the submitted fitting code.
        if gap>6: gap-=12
        elif gap< -6: gap+=12
        if not dd and rel>=-168:
            sd_sum+=md; sx_sum+=mx; ns+=1
        if dd:
            if not release_seen:release_gap=gap;release_seen=True
            if rel<=168+1e-9:gap_sum+=gap;td_sum+=td;tx_sum+=tx;ng+=1
            phases=(read,d,v,x)
            for j in range(4):
                ph=phases[j]
                if seen:
                    jump=ph-prev[j]
                    if jump>np.pi:offsets[j]-=twopi
                    elif jump< -np.pi:offsets[j]+=twopi
                u=ph+offsets[j];prev[j]=ph
                sums[j,0]+=1; sums[j,1]+=rel;sums[j,2]+=u;sums[j,3]+=rel*rel;sums[j,4]+=rel*u
            seen=True
        ddot=twopi/td-p[10]*np.sin(v-d)+fd+p[14]*light*np.sin(theta-d)
        vdot=twopi/tv+kvd*np.sin(d-v)+fv+p[15]*p[14]*light*np.sin(theta-v)
        xdot=twopi/tx+p[13]*np.sin(read-x)
        if mode==5:ddot=twopi/td+fd+p[14]*light*np.sin(theta-d);vdot=0.
        if mode==4:xdot=0.
        d=(d+dt*ddot)%twopi;v=(v+dt*vdot)%twopi;x=(x+dt*xdot)%twopi
    out=np.empty(10)
    for j in range(4):
        n,s_t,s_y,s_tt,s_ty=sums[j]
        slope=(n*s_ty-s_t*s_y)/(n*s_tt-s_t*s_t)
        out[j]=twopi/slope if slope>0 else np.nan
    out[4]=release_gap;out[5]=gap_sum/ng;out[6]=td_sum/ng;out[7]=tx_sum/ng
    out[8]=sd_sum/max(1,ns);out[9]=sx_sum/max(1,ns)
    if mode==5:out[2]=np.nan;out[4]=np.nan;out[5]=np.nan
    if mode==4:out[3]=np.nan;out[7]=np.nan;out[9]=np.nan
    return out


@njit(cache=True)
def screening_score(inv,ex,restricted):
    """Observable terms and weights transcribed from submitted model_score."""
    score=0.
    for i in range(5):
        if not np.isfinite(inv[i,0]):return 1e30
        sigma=.08 if i<3 else .18;weight=600. if i<3 else 500.
        err=abs(inv[i,0]-BT[i]);tol=.14 if i<3 else .25
        score+=weight*(err/sigma)**2
        if err>tol:score+=3000000+1200000*(err-tol)**2
        if restricted:continue
        for j in (1,2):
            if not np.isfinite(inv[i,j]) or not np.isfinite(ex[i,j]):return 1e30
            if i<3:score+=1400*((ex[i,j]-inv[i,j])/.08)**2
            else:
                score+=700*((ex[i,j]-ET[i])/.16)**2
                split=ex[i,j]-inv[i,j] if i==3 else inv[i,j]-ex[i,j]
                if split<.70:score+=2500000+1200000*(.70-split)**2
        if i<3:
            split=max(abs(ex[i,1]-inv[i,1]),abs(ex[i,2]-inv[i,2]))
            if split>.22:score+=9000000+4000000*(split-.22)**2
        if not np.isfinite(ex[i,4]) or not np.isfinite(ex[i,5]):return 1e30
        for j in (4,5):score+=140*((ex[i,j]-GT[i])/GS[i])**2
        err=max(abs(ex[i,4]-GT[i]),abs(ex[i,5]-GT[i]));tol=.45 if i<3 else .70
        if err>tol:score+=200000+150000*(err-tol)**2
    if not inv[1,0]<inv[2,0]<inv[0,0]:score+=1000000
    if inv[2,0]-inv[1,0]<.09:score+=1000000+600000*(.09-inv[2,0]+inv[1,0])**2
    if inv[0,0]-inv[2,0]<.18:score+=1000000+600000*(.18-inv[0,0]+inv[2,0])**2
    if not inv[3,0]<inv[2,0]<inv[4,0]:score+=1000000
    score+=5000*((inv[0,0]-inv[2,0])-.27)**2+5000*((inv[2,0]-inv[1,0])-.11)**2
    if restricted:return score
    for j in (0,1,2):
        if not ex[1,j]<ex[2,j]<ex[0,j]:score+=1000000
        if ex[2,j]-ex[1,j]<.08:score+=3500+3000*(.08-ex[2,j]+ex[1,j])**2
        if ex[0,j]-ex[2,j]<.12:score+=3500+3000*(.12-ex[0,j]+ex[2,j])**2
    if ex[3,5]>=-1.20:score+=6000000+2000000*(ex[3,5]+1.20)**2
    for i in (0,1,2,4):
        if ex[i,5]<=0:score+=1000000+100000*ex[i,5]**2
    if ex[4,5]<1.40:score+=3500000+1600000*(1.40-ex[4,5])**2
    if ex[3,5]>=0:score+=8000000+2500000*ex[3,5]**2
    diff=ex[3,1]-ex[3,2]
    if diff<=0:score+=8000000+2500000*diff**2
    elif diff<.18:score+=1500000+800000*(.18-diff)**2
    for j in (1,2):
        if ex[3,j]<=ex[2,j]+.20:score+=5000+1000*(ex[2,j]+.20-ex[3,j])**2
        if ex[4,j]>=ex[2,j]-.20:score+=1000000+100000*(ex[4,j]-ex[2,j]+.20)**2
    return score


def legacy_final_score(inv,ex,restricted=False):
    """Submitted final diagnostic's shared observable terms (X-only term omitted)."""
    if not np.isfinite(inv[:,0]).all():return 1e30
    behavior=np.abs(inv[:,0]-BT).sum()
    contrast=abs((inv[0,0]-inv[2,0])-.27)+abs((inv[2,0]-inv[1,0])-.11)
    value=10*behavior+25*contrast
    if restricted:return float(value)
    if not np.isfinite(ex[:,[1,2,5]]).all():return 1e30
    split=np.max(np.abs(ex[:3,1:3]-inv[:3,1:3]));diff=ex[3,1]-ex[3,2]
    signs=sum([ex[0,5]<=0,ex[1,5]<=0,ex[2,5]<=0,ex[4,5]<=0,ex[3,5]>=0])
    value+=8*split+100*max(0,.05-diff)+20*(diff<=0)+8*max(0,ex[3,5]+.60)+4*signs+np.abs(ex[1:,5]-GT[1:]).sum()
    return float(value)


def final_score(inv,ex,restricted=False):
    """Carry the original shared observation penalties through final selection.

    Screening uses slope periods; final selection supplies the same score with
    fine-step, 14-day Lomb-Scargle periods. In particular the absolute T22/T26
    ex vivo D/V targets remain penalized. No parameter-distance penalty or
    architecture-specific intrinsic-D prior is added to favor an old vector.
    """
    return float(screening_score(inv,ex,restricted))


@njit(cache=True,parallel=True)
def batch(vectors,mode,restricted,dt):
    scores=np.empty(len(vectors))
    for k in prange(len(vectors)):
        inv=np.empty((5,10));ex=np.empty((5,10))
        for i in range(5):
            inv[i]=summary(vectors[k],mode,CONDITIONS[i,0],CONDITIONS[i,1],dt,False)
            if not restricted:ex[i]=summary(vectors[k],mode,CONDITIONS[i,0],CONDITIONS[i,1],dt,True)
        scores[k]=screening_score(inv,ex,restricted)
    return scores


def bounds(arch):
    b=dict(model.BOUNDS)
    b.update(gphi_s=(-5.,0.),gp_s=(0.,6.),gphi_x=(0.,5.),gp_x=(-6.,0.),
             g_d=(1.,40.),g_x=(1.,40.),g_v=(1.,40.),g_coupling=(-.5,.5))
    if arch=='joint-DV-period-X':b['g_v']=(0.,40.)
    return np.array([b[k] for k in model.FREE[arch]])


def to_vectors(z,arch):
    z=np.atleast_2d(z); b=bounds(arch)
    p=np.zeros((len(z),len(model.NAMES)));p[:,:3]=24.
    p[:,[model.NAMES.index(k) for k in model.FREE[arch]]]=b[:,0]+np.clip(z,0,1)*(b[:,1]-b[:,0])
    return p


def normalized(p,arch):
    b=bounds(arch);return np.clip((np.asarray(p)[[model.NAMES.index(k) for k in model.FREE[arch]]]-b[:,0])/(b[:,1]-b[:,0]),0,1)


def archive_vector():
    record=pd.read_csv(PARENT/'data/archived_parameters.csv').iloc[0];p=np.zeros(len(model.NAMES))
    mapping={'tau_v0':'tau_v','gphi_s':'gphi_d','gp_s':'gp_d'}
    for key in model.FREE['D-period-X']:p[model.NAMES.index(key)]=record[mapping.get(key,key)]
    return p


def calibrated_vectors(arch,objective):
    p=archive_vector()
    if arch=='V-period-X':p[16]=p[3];p[3]=0
    elif arch=='coupling-X':p[3]=0;p[17]=-.484
    elif arch=='D-period-H':p[18:22]=[1.9,-.378,.10,.10]
    base=to_vectors(normalized(p,arch),arch)[0]
    old=json.loads((PARENT/'data/architecture_selected_parameters.json').read_text())
    selected=[z for z in old if z['architecture']==arch and
              z['objective']==('behavior_only' if 'behavior_only' in objective else 'first_peak')]
    old_p=base if not selected else to_vectors(normalized(selected[0]['vector'],arch),arch)[0]
    # Four transferred candidates for every job; their origin is recorded.
    both=(base+old_p)/2
    return [base,old_p,both,to_vectors(np.full(len(model.FREE[arch]),.5),arch)[0]]


def rescore(p,arch,dt=DT_FINAL):
    inv=np.empty((5,10));ex=np.empty((5,10)); rows=[]
    for i,c in enumerate(ORDER):
        for isex,arr in [(False,inv),(True,ex)]:
            s=summary(p,model.ARCHITECTURES[arch],*model.SCHEDULES[c],dt,isex)
            modern,tr=model.simulate(arch,p,c,ex=isex,dt=dt)
            dd=tr[(tr[:,0]>=0)&(tr[:,0]<=336)]
            # Exact original discrete Lomb-Scargle grid for final selection.
            for j,col in [(0,4),(1,1),(2,2),(3,3)]:
                if not np.isfinite(dd[:,col]).all():s[j]=np.nan;continue
                freq,power=LombScargle(dd[:,0],dd[:,col],center_data=True,fit_mean=True).autopower(
                    minimum_frequency=1/30,maximum_frequency=1/18,samples_per_peak=20)
                s[j]=1/freq[np.argmax(power)]
            arr[i]=s
            rows.append(dict(condition=c,ex_vivo=isex,behavior_h=s[0],D_h=s[1],V_h=s[2],X_h=s[3],
                gap_release_folded_h=s[4],gap_7day_folded_arithmetic_h=s[5],
                gap_first_peak_h=modern['gap_first_peak_h'],gap_7day_full_circular_h=modern['gap_days1_7_circular_h'],
                behavior_refined_LS_h=model.refined_period(tr,4),D_refined_LS_h=model.refined_period(tr,1),
                V_refined_LS_h=model.refined_period(tr,2)))
    return inv,ex,rows


def fit_job(arch,obj,args,fp):
    out=ROOT/'tables'/f'fit_{arch}_{obj}.json'
    if args.resume and out.exists():
        old=json.loads(out.read_text())
        if old.get('fingerprint')==fp:
            print('Reusing verified completed fit:',arch,obj,flush=True);return old
    start=time.perf_counter();mode=model.ARCHITECTURES[arch];restricted='behavior_only' in obj
    n=len(model.FREE[arch]);rng=np.random.default_rng(SEED)
    z=np.column_stack([(rng.permutation(args.latin_samples)+rng.random(args.latin_samples))/args.latin_samples for _ in range(n)])
    scores=np.empty(len(z)); history=[]
    def score(zs):return batch(to_vectors(zs,arch),mode,restricted,DT)
    for lo in range(0,len(z),1024):
        hi=min(lo+1024,len(z));scores[lo:hi]=score(z[lo:hi])
        if hi%8192==0 or hi==len(z):print(f'{arch}/{obj}: global {hi}/{len(z)}, best {scores[:hi].min():.5g}, {time.perf_counter()-start:.0f}s',flush=True)
    rank=np.argsort(scores);warm=calibrated_vectors(arch,obj)
    wz=np.array([normalized(p,arch) for p in warm]);ws=score(wz)
    starts=list(z[rank[:max(0,args.local_starts-4)]])+list(wz)
    startscores=list(scores[rank[:max(0,args.local_starts-4)]])+list(ws)
    pool=[dict(vector=to_vectors(z[j],arch)[0].tolist(),screening_score=float(scores[j]),origin='latin') for j in rank[:args.finalists]]
    calls=len(z)+len(wz)
    for li,(x,bs) in enumerate(zip(starts,startscores)):
        x=x.copy();step=.08
        for it in range(args.local_iters):
            probes=np.repeat(x[None,:],2*n,axis=0)
            for j in range(n):probes[2*j,j]=min(1,x[j]+.001);probes[2*j+1,j]=max(0,x[j]-.001)
            vals=score(probes);calls+=2*n
            grad=np.array([(vals[2*j]-vals[2*j+1])/max(.002,probes[2*j,j]-probes[2*j+1,j]) for j in range(n)])
            norm=np.linalg.norm(grad)
            if not np.isfinite(norm) or norm<1e-12:break
            accepted=False;trialstep=step
            for _ in range(8):
                trial=np.clip(x-trialstep*grad/norm,0,1);val=score(trial[None,:])[0];calls+=1
                if val<bs:x=trial;bs=val;accepted=True;break
                trialstep*=.5
            step=min(trialstep*1.2,.20) if accepted else step*.5
            history.append(dict(local_start=li+1,iteration=it+1,score=float(bs),step=step,accepted=accepted))
            if step<1e-4:break
        pool.append(dict(vector=to_vectors(x,arch)[0].tolist(),screening_score=float(bs),origin='local'))
        if (li+1)%8==0:print(f'{arch}/{obj}: local {li+1}/{len(starts)}',flush=True)
    # Include all four transferred candidates as the manuscript did for its
    # calibrated candidate. Every final candidate is rescored identically.
    shortlist=sorted(pool,key=lambda q:q['screening_score'])[:args.finalists]
    shortlist += [dict(vector=p.tolist(),screening_score=float(s),origin=f'transferred_{j+1}') for j,(p,s) in enumerate(zip(warm,ws))]
    unique={tuple(q['vector']):q for q in shortlist};shortlist=list(unique.values())
    finalists=[]
    for j,q in enumerate(shortlist):
        inv,ex,rows=rescore(np.array(q['vector']),arch)
        q.update(final_diagnostic_score=final_score(inv,ex,restricted),
                 fine_screening_score=float(screening_score(inv,ex,restricted)),endpoints=rows)
        finalists.append(q)
        if (j+1)%40==0:print(f'{arch}/{obj}: final LS {j+1}/{len(shortlist)}',flush=True)
    best=min(finalists,key=lambda q:q['final_diagnostic_score'])
    result=dict(architecture=arch,objective=obj,seed=SEED,free_parameter_count=n,fingerprint=fp,
                selection_criterion='shared_observable_penalty_v2',
                selection_description='Same shared observable screening penalties evaluated on fine-step discrete Lomb-Scargle endpoints; absolute ex vivo T-cycle targets retained.',
                selected=best,vector=best['vector'],parameters={k:best['vector'][model.NAMES.index(k)] for k in model.FREE[arch]},
                final_diagnostic_score=best['final_diagnostic_score'],evaluations_screening=calls,
                latin_samples=len(z),local_starts=len(starts),maximum_local_iterations=args.local_iters,
                final_candidates=len(finalists),elapsed_seconds=time.perf_counter()-start,
                bounds={k:list(b) for k,b in zip(model.FREE[arch],bounds(arch))},
                selected_origin=best['origin'],history=history,finalists=finalists)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'COMPLETE {arch}/{obj}: final {best["final_diagnostic_score"]:.6g}, {result["elapsed_seconds"]:.0f}s',flush=True)
    return result

