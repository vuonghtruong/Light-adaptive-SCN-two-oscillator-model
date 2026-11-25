"""Graphical abstract from the same selected-model trajectories as Figure 3.

An open circle or a bar marks the receiving end of an attractive or repulsive
SCN connection. Arrowheads are reserved for light input, memory and feedback.
"""
from common import *
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

BROWN='#795438'
IVORY='#F3EEE7'
LIGHT='#F4D57C'
GOLD='#B88716'


def diagram(ax):
    ax.set(xlim=(0,100),ylim=(0,77)); ax.set_aspect('equal'); ax.axis('off')
    ax.text(50,75,'SCN with systemic feedback',ha='center',va='top',fontsize=9.5,fontweight='bold')
    ax.add_patch(FancyBboxPatch((8,26),67,34,boxstyle='round,pad=0,rounding_size=3',
        facecolor='#F7F8FA',edgecolor='#9299A1',lw=1.35,linestyle=(0,(4,3))))
    ax.text(12,55,'SCN',fontsize=8.5,color='#5B626A',fontweight='bold')
    d=np.array([26.,42.]);v=np.array([62.,42.]);x=np.array([79.,14.]);r=6.2
    # Coupling endpoints identify the receiving population, consistent with K_ij.
    upper=46.;lower=38.;left=35.;right=53.
    ax.plot([left,right],[upper,upper],color=REG['V'],lw=2.2,solid_capstyle='butt')
    ax.plot([left,left],[upper-2.9,upper+2.9],color=REG['V'],lw=2.2,solid_capstyle='butt')
    ax.text(47,51,r'$K_{DV}$ repulsive',color=REG['V'],ha='center',fontsize=8.5)
    circle_radius=1.55
    ax.plot([left,right-circle_radius],[lower,lower],color=REG['D'],lw=2.2,solid_capstyle='butt')
    ax.add_patch(Circle((right,lower),circle_radius,fc='white',ec=REG['D'],lw=2.2,zorder=4))
    ax.text(44,31.3,r'$K_{VD}$ attractive',color=REG['D'],ha='center',fontsize=8.5)

    # Direct photic phase forcing is stronger in V; X receives systemic feedback.
    sun=np.array([86.,62.]);sr=4.9
    ax.add_patch(Circle(sun,sr,fc=LIGHT,ec=GOLD,lw=1.5))
    for angle in np.arange(0,2*np.pi,np.pi/4):
        u=np.array([np.cos(angle),np.sin(angle)])
        a=sun+u*(sr+1.5);b=sun+u*(sr+3.1)
        ax.plot([a[0],b[0]],[a[1],b[1]],color=GOLD,lw=1.4)
    ax.text(86,72.5,'Light',ha='center',fontsize=9,fontweight='bold')
    ax.add_patch(FancyArrowPatch((80,61),(28,49),arrowstyle='-|>',mutation_scale=11,
        connectionstyle='arc3,rad=.15',color=GOLD,lw=1.4))
    ax.add_patch(FancyArrowPatch((82,57),(67,48),arrowstyle='-|>',mutation_scale=12,
        color=GOLD,lw=2.6))
    ax.text(86,48.5,'V input\n> D input',ha='center',va='top',fontsize=8,color='#8A6500',linespacing=1.3)

    # Distinct arrowed systemic connections; the circle/bar convention is for D-V.
    ax.add_patch(FancyArrowPatch((74.7,18.5),(71.5,29),arrowstyle='-|>',mutation_scale=12,
        connectionstyle='arc3,rad=.18',color=REG['X'],lw=2.0))
    ax.add_patch(FancyArrowPatch((66.5,27),(72.8,13),arrowstyle='-|>',mutation_scale=12,
        connectionstyle='arc3,rad=.24',color='#737A82',lw=1.8))
    ax.text(55,18,'Systemic\nfeedback',ha='center',va='center',fontsize=8,linespacing=1.3,color='#40464D')
    ax.text(79,3.4,'No direct photic\nphase forcing',ha='center',va='top',fontsize=8,color='#5B626A',linespacing=1.25)

    ax.add_patch(FancyBboxPatch((7,6),35,13,boxstyle='round,pad=0,rounding_size=2',
        fc=IVORY,ec=BROWN,lw=1.5))
    ax.text(24.5,12.5,r'History $M_D\to\tau_D$',ha='center',va='center',fontsize=8.5,color=BROWN)
    ax.add_patch(FancyArrowPatch((24.5,20),(25.5,34.7),arrowstyle='-|>',mutation_scale=12,color=BROWN,lw=1.8))
    # X also retains an adaptive history state. The annotation names its effect
    # without depicting the schedule/phase acquisition proxy as an SCN-only drive.
    ax.add_patch(FancyBboxPatch((77,28),22,13,boxstyle='round,pad=0,rounding_size=2',
        fc=IVORY,ec=BROWN,lw=1.5))
    ax.text(88,34.5,'History\n'+r'$M_X\to\tau_X$',ha='center',va='center',
        fontsize=8.5,color=BROWN,linespacing=1.25)
    ax.add_patch(FancyArrowPatch((88,27),(82,20.2),arrowstyle='-|>',mutation_scale=12,
        color=BROWN,lw=1.8))
    for label,xy in [('D',d),('V',v),('X',x)]:
        ax.add_patch(Circle(xy,r,fc=REG[label],ec='white',lw=1.2,zorder=5))
        ax.text(*xy,label,ha='center',va='center',fontsize=12,color='white',fontweight='bold',zorder=6)


def main():
    traces=read('full_model_dynamic_traces.csv.gz')
    endpoints=pd.read_csv(TABLES/'full_model_dynamic_endpoints.csv')
    provenance=json.loads((ROOT/'validation/full_model_dynamic_data.json').read_text())
    import hashlib
    assert provenance['trace_sha256']==hashlib.sha256((DATA/'full_model_dynamic_traces.csv.gz').read_bytes()).hexdigest()
    f=fig(146)
    # The lower schematic and summaries share one baseline; generous outside margins.
    f.text(.025,.974,'A',fontweight='bold',fontsize=11,va='top')
    ent=f.add_axes([.08,.748,.265,.155])
    inv=f.add_axes([.43,.824,.54,.106])
    exv=f.add_axes([.43,.644,.54,.106])
    row=traces[traces.condition.eq('T22')]
    shade=[];trace_checks=[]
    for ax,context,lo,hi,keys in [(ent,'invivo',-72,0,['D','V','X']),
            (inv,'invivo',0,168,['D','V','X']),(exv,'exvivo',0,168,['D','V'])]:
        q=row[row.context.eq(context)&row.time_from_DD_h.between(lo,hi)]
        t=q.time_from_DD_h.to_numpy()/24
        for key in keys:
            y=(q[key].to_numpy()+1)/2
            line,=ax.plot(t,y,color=REG[key],lw=1.7 if key!='X' else 1.15,
                ls=':' if key=='X' else '-',zorder=3 if key=='X' else 2)
            assert np.array_equal(line.get_xdata(),t) and np.array_equal(line.get_ydata(),y)
            trace_checks.append(dict(context=context,region=key,range_h=[lo,hi],points=len(q),exact_coordinate_match=True))
        ax.set(ylim=(-.04,1.04),yticks=[0,1],xlim=(lo/24,hi/24))
    # Integrator uses time since entrainment start: t_absolute = t_from_DD + 720.
    for k in range(28,34):
        start=max(-72.,k*22.-720.);end=min(0.,k*22.+11.-720.)
        if end>start:
            ent.axvspan(start/24,end/24,fc=LIGHT,alpha=.63,lw=0,zorder=0);shade.append([start,end])
    ent.set(xticks=[-3,-2,-1,0],xlabel='Day from DD',ylabel='Normalized activity')
    ent.set_title('T22 entrainment',fontsize=9,fontweight='bold',pad=5)
    inv.set(xticks=[0,2,4,6]);inv.tick_params(labelbottom=False)
    inv.set_title('In vivo DD: feedback retained',fontsize=9,fontweight='bold',pad=5)
    exv.set(xticks=[0,2,4,6],xlabel='DD day')
    exv.set_title('Ex vivo: X disconnected',fontsize=9,fontweight='bold',pad=5)
    for y in [.867,.704]:
        f.add_artist(ConnectionPatch((.357,.812),(.414,y),'figure fraction','figure fraction',
            arrowstyle='-|>',mutation_scale=11,lw=1.5,color='#626A73'))
    handles=[Line2D([0],[0],color=REG[k],lw=1.7 if k!='X' else 1.15,ls=':' if k=='X' else '-',label=k) for k in ['D','V','X']]
    handles.append(Patch(fc=LIGHT,label='Light'))
    f.legend(handles=handles,loc='upper right',bbox_to_anchor=(.98,1.012),ncol=4,frameon=False,
        fontsize=8,handlelength=1.4,columnspacing=1.15,handletextpad=.45)
    f.text(.025,.554,'B',fontweight='bold',fontsize=11,va='top')
    ax=f.add_axes([.025,.033,.575,.506]);diagram(ax)
    f.text(.649,.554,'C',fontweight='bold',fontsize=11,va='top')
    period_records=[]
    configs=[(.329,['LP','LD','SP'],['16:8 LP','12:12 LD','8:16 SP'],'Photoperiod'),
             (.075,['T22','LD','T26'],['T22','T24','T26'],'T-cycle')]
    for bottom,conditions,labels,title in configs:
        a=f.add_axes([.72,bottom,.25,.179])
        for key,context,color,marker,label in [('behavior_h',False,'#60676F','o','Behavior, in vivo'),
                ('D_h',True,REG['D'],'s','D, ex vivo'),('V_h',True,REG['V'],'^','V, ex vivo')]:
            values=[float(endpoints[endpoints.condition.eq(c)&endpoints.ex_vivo.eq(context)].iloc[0][key]) for c in conditions]
            line,=a.plot(range(3),values,color=color,marker=marker,lw=1.7,ms=4.5,mew=.7,label=label)
            assert np.array_equal(line.get_ydata(),values)
            period_records.extend(dict(condition=c,ex_vivo=context,column=key,period_h=y) for c,y in zip(conditions,values))
        a.set(xlim=(-.13,2.13),ylim=(22,26),yticks=[22,24,26],xticks=range(3),xticklabels=labels,ylabel='Period (h)')
        a.set_title(title,fontsize=9,fontweight='bold',pad=5)
        a.grid(False,which='both',axis='both')
        if title=='Photoperiod':
            a.legend(loc='upper left',fontsize=8,frameon=False,handlelength=1.6,handletextpad=.5,borderaxespad=.3,labelspacing=.3)
    prepare_figure(f)
    for a in [ent,inv,exv]+f.axes[-2:]:
        for spine in a.spines.values():spine.set_linewidth(1.15)
        a.tick_params(width=1.05,pad=2.5)
    # Keep the nearly overlapping X trace fine and dotted so D/V remain visible.
    for a in [ent,inv]:a.lines[-1].set_linewidth(1.15)
    out=ROOT/'figures/graphical_abstract';out.mkdir(parents=True,exist_ok=True)
    for ext in ['svg','pdf','png']:f.savefig(out/f'Graphical_abstract.{ext}',dpi=600,facecolor='white')
    record_publication_style(f,'Graphical_abstract')
    style_path=ROOT/'validation/publication_style_checks.json';styles=json.loads(style_path.read_text())
    styles['Graphical_abstract'].update(axes_linewidth_pt=1.15,trace_linewidth_pt=1.7,X_trace_linewidth_pt=1.15,coupling_linewidth_pt=2.2)
    style_path.write_text(json.dumps(styles,indent=2)+'\n',encoding='utf-8')
    assert not styles['Graphical_abstract']['text_outside_canvas'],styles['Graphical_abstract']['text_outside_canvas']
    report=dict(passed=True,date='2026-10-18',dimensions_mm=[175,146],PNG_dpi=600,
        source='Current full-model trajectories and dynamic endpoint table used in Figure 3; original July schematic curves superseded.',
        trace_sha256=provenance['trace_sha256'],endpoint_sha256=hashlib.sha256((TABLES/'full_model_dynamic_endpoints.csv').read_bytes()).hexdigest(),
        trace_checks=trace_checks,period_records=period_records,light_intervals_relative_DD_h=shade,
        conventions=dict(repulsive='V to D: straight bar at D/left receiving end',attractive='D to V: open circle at V/right receiving end'),
        labels=dict(SP='8:16 SP',LP='16:8 LP',LD='12:12 LD',X='No direct photic phase forcing',
                    dorsal_history='M_D -> tau_D',systemic_history='M_X -> tau_X'),
        period_summary_gridlines_visible=[sum(line.get_visible() for line in a.get_xgridlines()+a.get_ygridlines()) for a in f.axes[-2:]],
        period_estimator='Refined estimates over the complete 14-day DD record, as in Figure 3',
        fitted_parameters_changed=False,model_reintegrated=False,experimental_error_bars=False)
    (ROOT/'validation/graphical_abstract_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    plt.close(f);print('Saved Graphical_abstract',flush=True)


if __name__=='__main__':main()
