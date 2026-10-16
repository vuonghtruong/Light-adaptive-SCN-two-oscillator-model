"""Reference Figure 3 and prediction Figure 4, using one full-model renderer."""
from common import *
from actogram_style import doubleplot_axis,dynamic_legend

OUTPUT_COLORS={'behavior_h':'#666666','D_h':REG['D'],'V_h':REG['V'],'X_h':REG['X']}
OUTPUT_LABELS={'behavior_h':'Behavior','D_h':'D','V_h':'V','X_h':'X'}

def full_data():
    p=DATA/'full_model_dynamic_traces.csv.gz'
    if not p.exists():
        from full_model_data import main
        main()
    return read('full_model_dynamic_traces.csv.gz'),pd.read_csv(TABLES/'full_model_dynamic_endpoints.csv')

def calibration_mark(ax,x,y,sem=None):
    """Draw only recorded SEM; a white underlay separates targets from model points."""
    caps=2.5 if sem is not None else 0
    # Draw the complete underlay first, so white cap halos cannot obscure the
    # black vertical stem when a recorded SEM is small on the shared axis.
    ax.errorbar(x,y,yerr=sem,fmt='_',ms=9.2,capsize=caps,color='white',
                lw=2.7,elinewidth=2.7,capthick=2.7,zorder=8)
    return ax.errorbar(x,y,yerr=sem,fmt='_',ms=8,capsize=caps,color='black',
                       lw=1.05,elinewidth=1.05,capthick=1.05,zorder=9)


def period_summary(ax,ref,ensemble,conditions_,ex,keys,target=None,ensemble_color='black'):
    rng=np.random.default_rng(2241);width=.76/len(keys);audit=[]
    for j,key in enumerate(keys):
        # Fail visibly if a requested endpoint column is absent rather than omit points.
        if key not in ensemble.columns:raise KeyError(f'Missing ensemble output: {key}')
        for i,c in enumerate(conditions_):
            r=ref[ref.condition.eq(c)&ref.ex_vivo.eq(ex)].iloc[0]
            x=i+(j-(len(keys)-1)/2)*width
            ax.bar(x,r[key]-22,bottom=22,width=width*.85,color=OUTPUT_COLORS[key],alpha=.85,lw=0,zorder=1)
            e=ensemble[ensemble.condition.eq(c)&ensemble.ex_vivo.eq(ex)]
            values=e[key].to_numpy();finite=np.isfinite(values)
            # Jitter x only: preserve the small, genuine behavioral-period spread.
            jitter=rng.uniform(-width*.30,width*.30,len(e))
            ax.scatter(x+jitter[finite],values[finite],s=6.2 if key=='behavior_h' else 4.8,
                       color=ensemble_color or 'black',alpha=.32 if key=='behavior_h' else .26,
                       linewidths=0,zorder=4)
            ax.plot(x,r[key],'o',ms=3.7,mfc='white',mec='black',mew=.8,zorder=7)
            audit.append(dict(condition=c,ex_vivo=bool(ex),source_column=key,
                              available_rows=len(e),drawn_points=int(finite.sum()),
                              minimum_h=float(values[finite].min()),maximum_h=float(values[finite].max()),
                              horizontal_jitter_only=True,point_color=ensemble_color or 'black'))
    if target is not None:
        for i,c in enumerate(conditions_):
            row=target.loc[c];key='exvivo_scn_target_h' if ex else 'behavior_mean_h'
            # Behavioral SEM is recorded; no ex-vivo SEM is supplied in this array.
            sem=None if ex else float(row.behavior_display_sem_h)
            calibration_mark(ax,i if ex else i+.30,row[key],sem)
    values=list(ref[ref.condition.isin(conditions_)&ref.ex_vivo.eq(ex)][keys].to_numpy().ravel())
    values+=list(ensemble[ensemble.condition.isin(conditions_)&ensemble.ex_vivo.eq(ex)][keys].to_numpy().ravel())
    if target is not None:
        for c in conditions_:
            val=target.loc[c,'exvivo_scn_target_h' if ex else 'behavior_mean_h'];err=0 if ex else target.loc[c,'behavior_display_sem_h']
            values.extend([val-err,val+err])
    low,high=np.nanmin(values),np.nanmax(values);pad=max(.16,(high-low)*.13)
    ax.set(xticks=range(len(conditions_)),xticklabels=[c.replace(' LD','') for c in conditions_],ylabel='Ex vivo period (h)' if ex else 'In vivo period (h)',xlim=(-.5,len(conditions_)-.5),ylim=(low-pad,high+pad))
    ax._ensemble_display=audit


def write_summary_display(name,axes):
    path=ROOT/'validation/main_summary_display.json'
    audit=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    audit[name]=dict(period_ensemble_points=[r for ax in axes for r in getattr(ax,'_ensemble_display',[])],
                     primary_marker='white-filled black-outline circle',
                     target_style='black with white outline; recorded SEM only',
                     model_uncertainty='64 deterministic initial-phase simulations, not experimental SEM')
    path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')


def phase_summary(ax,ref,ensemble,conditions_,target=None,point_color=None):
    rng=np.random.default_rng(2241)
    for i,c in enumerate(conditions_):
        r=ref[ref.condition.eq(c)&ref.ex_vivo].iloc[0]
        e=ensemble[ensemble.condition.eq(c)&ensemble.ex_vivo]
        ax.scatter(i+rng.uniform(-.065,.065,len(e)),e.gap_days1_7_circular_h,s=3,color=point_color or COL[c],alpha=.18 if point_color else .23)
        ax.plot(i,r.gap_days1_7_circular_h,'o',ms=3.8,color=point_color or COL[c])
        if target is not None:calibration_mark(ax,i+.22,target.loc[c,'gap_mean_h'],float(target.loc[c,'gap_sem_h']))
    ax.axhline(0,color='#777777',lw=.5,zorder=0)
    ax.set(xticks=range(len(conditions_)),xticklabels=[c.replace(' LD','') for c in conditions_],ylabel='7-day D–V gap (h)',xlim=(-.5,len(conditions_)-.5))

def summary_legend(f,bbox,keys=None):
    keys=list(OUTPUT_COLORS) if keys is None else keys
    h=[Patch(facecolor=OUTPUT_COLORS[k],label=OUTPUT_LABELS[k],alpha=.85) for k in keys]
    return f.legend(handles=h,loc='lower center',bbox_to_anchor=bbox,ncol=4,frameon=False,fontsize=6,handlelength=1.3,columnspacing=1)

def trajectory_pairs(f,spec,tr,conditions_,labels,wspace=.21):
    columns=spec.subgridspec(1,len(conditions_),wspace=wspace)
    for i,c in enumerate(conditions_):
        pair=columns[i].subgridspec(2,1,hspace=.14)
        for ex in [False,True]:
            ax=f.add_subplot(pair[int(ex)])
            doubleplot_axis(ax,tr,c,ex,show_ylabels=i==0,show_xlabel=ex,show_condition=not ex)
            if not ex:letter(ax,labels[i],x=-.18,y=1.035)
            if i==0:
                ax.text(-.36,.50,'Ex vivo' if ex else 'In vivo',transform=ax.transAxes,
                        ha='center',va='center',fontsize=6.5,rotation=90,fontstyle='italic')


def explant_sensitivity():
    from explant_robustness import build
    data,summary,baseline=build()
    return data,baseline


def robustness_panels(axes,ref,order):
    z=ref[ref.ex_vivo].set_index('condition')
    axes[0].scatter(range(5),[z.loc[c,'locking_ratio_release'] for c in order],c=[COL[c] for c in order],s=18)
    axes[0].axhline(1,color='black',ls='--',lw=.65)
    axes[0].set(xticks=range(5),xticklabels=order,ylabel=r'$|\Delta\omega/K_{\rm eff}|$',xlim=(-.5,4.5))
    data,baseline=explant_sensitivity()
    for c in ['T22','LD','T26']:
        for signal,mark,ls in [('D','o','-'),('V','^','--')]:
            q=data[data.test.eq('coupling')&data.region.eq(signal)].sort_values('coupling_percent')
            axes[1].plot(q.coupling_percent,q[c+'_period_h'],color=COL[c],marker=mark,ms=2.8,ls=ls,lw=.8)
    axes[1].set(xlabel='D–V coupling (% reference)',ylabel='Ex vivo period (h)',xticks=[50,100,150])
    summary=[]
    for c in ['T22','LD','T26']:
        for signal,mark,ls,offset in [('D','o','-',-1.5),('V','^','--',1.5)]:
            points=[]
            for ret in [0,50,100]:
                q=data[data.test.eq('memory_phase')&data.region.eq(signal)&data.memory_retained_percent.eq(ret)]
                values=q[c+'_period_h'];center=float(values.mean());lo=float(values.min());hi=float(values.max())
                points.append((ret,center,lo,hi))
                summary.append(dict(condition=c,region=signal,memory_retained_percent=ret,mean_period_h=center,min_period_h=lo,max_period_h=hi,perturbation_combinations=len(q)))
            p=np.asarray(points)
            axes[2].errorbar(p[:,0]+offset,p[:,1],yerr=np.stack([p[:,1]-p[:,2],p[:,3]-p[:,1]]),color=COL[c],marker=mark,ms=2.8,ls=ls,lw=.8,elinewidth=.55,capsize=1.6,alpha=.90)
    pd.DataFrame(summary).to_csv(TABLES/'Figure3K_absolute_period_summary.csv',index=False)
    axes[2].set(xlabel='D memory retained (%)',ylabel='Ex vivo period (h)',xticks=[0,50,100],xlim=(-8,108))
    limits=data[['T22_period_h','LD_period_h','T26_period_h']].to_numpy()
    low,high=float(limits.min()),float(limits.max());pad=max(.15,(high-low)*.08)
    for ax in axes[1:]:ax.set_ylim(low-pad,high+pad)


def main3():
    tr,ref=full_data();ens=read('perturbation_initial_ensemble.csv');target=plotting_targets()
    order=['LD','SP','LP','T22','T26']
    f=fig(246);gs=f.add_gridspec(3,1,left=.105,right=.985,top=.956,bottom=.115,height_ratios=[2.13,.70,.70],hspace=.29)
    trajectory_pairs(f,gs[0],tr,order,'ABCDE')
    middle=gs[1].subgridspec(1,3,wspace=.43);axes=[f.add_subplot(middle[i]) for i in range(3)]
    period_summary(axes[0],ref,ens,order,False,['behavior_h'],target)
    axes[0].set_ylabel('Behavioral period (h)')
    period_summary(axes[1],ref,ens,order,True,['D_h','V_h'],target,ensemble_color='black')
    phase_summary(axes[2],ref,ens,order,target)
    axes[1].legend(handles=[Patch(facecolor=REG['D'],label='D'),Patch(facecolor=REG['V'],label='V')],loc='upper left',frameon=False,ncol=2,fontsize=5.7,handlelength=.9,columnspacing=.8)
    for ax,label in zip(axes,'FGH'):letter(ax,label,x=-.20)
    write_summary_display('Figure_3',axes)
    bottom=gs[2].subgridspec(1,3,wspace=.43);axes=[f.add_subplot(bottom[i]) for i in range(3)]
    robustness_panels(axes,ref,order)
    for ax,label in zip(axes,'IJK'):letter(ax,label,x=-.20)
    axes[1].legend(handles=[Line2D([],[],marker='o',ls='-',color='#555555',label='D'),Line2D([],[],marker='^',ls='--',color='#555555',label='V')],frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.55,1.0),fontsize=5.3,handlelength=1.3,columnspacing=.8)
    schedule_legend(figobj=f,bbox=(.54,.049),order=order)
    dynamic_legend(f,bbox=(.54,.018))
    save(f,'Figure_3')

def main4():
    tr,ref=full_data();ens=read('original_logic_prediction_ensemble.csv');future=['18:6 LD','T23','T25']
    f=fig(196);gs=f.add_gridspec(2,1,left=.105,right=.98,top=.956,bottom=.13,height_ratios=[2.25,.8],hspace=.27)
    trajectory_pairs(f,gs[0],tr,future,'ABC',wspace=.28)
    low=gs[1].subgridspec(1,3,wspace=.42);a=[f.add_subplot(low[i]) for i in range(3)]
    period_summary(a[0],ref,ens,future,False,['behavior_h'])
    a[0].set_ylabel('Behavioral period (h)')
    period_summary(a[1],ref,ens,future,True,['D_h','V_h'],ensemble_color='black')
    phase_summary(a[2],ref,ens,future,point_color='black')
    for ax,label in zip(a,'DEF'):letter(ax,label,x=-.20)
    write_summary_display('Figure_4',a)
    summary_legend(f,(.53,.052),keys=['behavior_h','D_h','V_h']);dynamic_legend(f,bbox=(.53,.019))
    save(f,'Figure_4')

if __name__=='__main__':main3();main4()
