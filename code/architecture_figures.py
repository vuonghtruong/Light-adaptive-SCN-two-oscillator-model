from common import *
import architecture_model as model

def comparison_rows(axes,architectures=['D-period-X','coupling-X'],residual=False):
 target=plotting_targets()
 for ai,arch in enumerate(architectures):
  z=selected_endpoints(arch);off=(ai-(len(architectures)-1)/2)*.23;mark=['o','s','^'][ai]
  for ax,(ctx,col,tkey,se,ylabel) in zip(axes,[(False,'behavior_h','behavior_mean_h','behavior_display_sem_h','Behavior (h)'),(True,'behavior_h','exvivo_scn_target_h',None,'Ex vivo SCN (h)'),(True,'gap_7day_folded_arithmetic_h','gap_mean_h','gap_sem_h','D–V gap (h)')]):
   for i,c in enumerate(ORDER):
    r=z[(z.condition==c)&z.ex_vivo.eq(ctx)].iloc[0];v=r[col]-target.loc[c,tkey] if residual else r[col]
    if tkey=='exvivo_scn_target_h':
     baseline=target.loc[c,tkey] if residual else 0
     ax.plot([i+off,i+off],[r.D_h-baseline,r.V_h-baseline],color=COL[c],lw=.8,alpha=.75)
     if residual:
      ax.plot(i+off-.025,r.D_h-baseline,marker=mark,ls='',ms=2.7,mfc=COL[c] if ai==0 else 'white',mec=COL[c])
      ax.plot(i+off+.025,r.V_h-baseline,marker=mark,ls='',ms=2.7,mfc=COL[c] if ai==0 else 'white',mec=COL[c])
    ax.plot(i+off,v,marker=mark,ls='',ms=3.5,mfc=COL[c] if ai==0 else 'white',mec=COL[c],mew=.8)
    if ai==0:
     ax.errorbar(i,0 if residual else target.loc[c,tkey],yerr=None if se is None else target.loc[c,se],fmt='_',ms=6,color='black',capsize=1.5,lw=.5)
   conditions(ax,('Residual (h)' if residual else ylabel));ax.tick_params(labelsize=5.7)



def sensitivity_lines(ax,data,xkey,ykey,ls='-',phase=False,selected=True):
 """Connect full forward simulations in gain order, never across a wrap cut."""
 for c in ORDER:
  z=data[data.condition==c].sort_values('gain_scale')
  x=z[xkey].to_numpy();y=z[ykey].to_numpy().copy()
  y[~np.isfinite(y)]=np.nan
  cuts=np.abs(np.diff(y))>12 if phase else np.zeros(len(y)-1,dtype=bool)
  if ykey=='release_locking_ratio':cuts|=z.release_K_eff_rad_h.to_numpy()[:-1]*z.release_K_eff_rad_h.to_numpy()[1:]<0
  for ids in np.split(np.arange(len(z)),np.flatnonzero(cuts)+1):
   ax.plot(x[ids],y[ids],color=COL[c],ls=ls,lw=.85)
  if selected:
   r=z[np.isclose(z.gain_scale,1)].iloc[0]
   ax.plot(r[xkey],r[ykey],marker='o',ms=3.7,ls='',mfc=COL[c],mec='black',mew=.45,zorder=5)


def memory_mechanism(ax):
 """Equivalent decomposition of the implemented single dorsal memory state.

 Q_D and P_D start at zero and use the same linear filter, so their sum obeys
 the original M_D equation exactly. Chi gates the exposure interval, including
 its dark phases; it is zero only after release to DD. No separate states are
 added to the simulator by this explanatory diagram.
 """
 from matplotlib.patches import Ellipse,Rectangle
 ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 green='#267326';purple='#9B218E'
 def label(x,y,text,**kwargs):
  return ax.text(x,y,text,ha='center',va='center',fontsize=6.3,**kwargs)
 def arrow(a,b,color='#333333',width=.85,style='-|>'):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=7,lw=width,color=color))
 label(.5,.978,'Light schedule  '+r'$I(t),\ \theta(t)$',fontweight='bold')
 label(.5,.925,r'$\chi=1$ during entrainment; $\chi=0$ in DD')
 for x,color in [(0,'#EBF6EC'),(.515,'#EBF1FC')]:
  ax.add_patch(Rectangle((x,.710),.485,.165,facecolor=color,edgecolor='none'))
 label(.2425,.85,'Photoperiod',fontweight='bold')
 label(.7575,.85,'Phase mismatch',fontweight='bold')
 label(.2425,.790,r'$u_P=g_{P,D}\left[\frac{1+\sin\phi_D}{2}-I\right]$')
 label(.7575,.790,r'$u_\phi=g_{\phi,D}I\sin(\theta-\phi_D)$')
 label(.2425,.735,r'$T_m\dot Q_D=\chi u_P-Q_D$')
 label(.7575,.735,r'$T_m\dot P_D=\chi u_\phi-P_D$')
 arrow((.245,.708),(.425,.645));arrow((.755,.708),(.575,.645))
 label(.5,.625,r'$M_D=P_D+Q_D$',fontweight='bold')
 label(.5,.582,r'$T_m=240\ \mathrm{h}$; zero memory at exposure onset')
 arrow((.44,.563),(.255,.523),green);arrow((.56,.563),(.745,.523),purple)
 label(.25,.5,'Adaptive period  ●',color=green,fontweight='bold')
 label(.75,.5,'Adaptive coupling  □',color=purple,fontweight='bold')
 label(.25,.452,r'$\tau_D=\tau_{D,0}-g_D\tanh M_D$')
 label(.75,.452,r'$K_{VD}=K_{VD,0}+g_K\tanh M_D$')
 label(.25,.405,r'$K_{VD},K_{DV},\tau_V$ fixed')
 label(.75,.405,r'$\tau_D,\tau_V,K_{DV}$ fixed')
 xy={'D':(.25,.280),'V':(.75,.280),'X':(.50,.130)}
 for name,(x,y) in xy.items():
  ax.add_patch(Ellipse((x,y),.115,.065,facecolor='white',edgecolor='#555555',lw=.85))
  label(x,y,name,fontweight='bold')
 # The adaptive K_VD term acts from D to V; the reverse K_DV stays fixed.
 arrow((.315,.296),(.685,.296),'#777777',style='<-')
 arrow((.315,.264),(.685,.264),purple)
 arrow((.25,.376),(.25,.317),green)
 arrow((.73,.376),(.54,.266),purple)
 for left,right in [('D','X'),('V','X')]:
  a=np.array(xy[left]);b=np.array(xy[right]);d=(b-a)/np.linalg.norm(b-a)
  arrow(a+.054*d,b-.054*d,'#777777',.65,'<->')
 label(.5,.067,'Same adaptive X in both architectures')
 label(.5,.024,'Period bounds: 16–34 h')


def validate_memory_decomposition():
 """Check the displayed split against every integration step of live histories."""
 from scipy.signal import lfilter
 import reference_model as full
 dt=1/64;tm=full.PARAMS['mem_tau_d'];checks=[]
 for condition in ORDER:
  a=full.simulate(condition,dt=dt)
  cycle,photo=full.SCHEDULES[condition];time=a[:,0]+720.
  chi=(a[:,0]<0).astype(float)
  light=chi*((time%cycle)<photo)
  theta=np.where(chi>0,2*np.pi*(time%cycle)/cycle,0.)
  up=full.PARAMS['gp_d']*((1+np.sin(a[:,5]))/2-light)
  uphi=full.PARAMS['gphi_d']*light*np.sin(theta-a[:,5])
  q=lfilter([dt/tm],[1,-(1-dt/tm)],chi*up)
  p=lfilter([dt/tm],[1,-(1-dt/tm)],chi*uphi)
  checks.append(dict(condition=condition,integration_steps=len(a),
                     maximum_memory_difference=float(np.max(abs(p+q-a[:,8]))),
                     maximum_drive_difference=float(np.max(abs(chi*(up+uphi)-a[:,19])))))
 error=max(r['maximum_memory_difference'] for r in checks)
 report=dict(passed=error<1e-11 and all(r['maximum_drive_difference']<1e-11 for r in checks),
             maximum_memory_difference=error,integration_step_h=dt,memory_timescale_h=tm,
             zero_initial_component_states=True,simulation_model_unchanged=True,
             explanation='P_D and Q_D are an exact linear-filter decomposition of M_D for any phase history, not additional simulated mechanisms. Chi is one throughout the exposure schedule and zero in DD.',
             cases=checks)
 (ROOT/'validation/Figure2_memory_mechanism.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 return {'test':'Figure 2 displayed memory decomposition equals the original integrated M_D',
         'passed':report['passed'],'maximum_memory_difference':error}


def main2():
 from architecture_sensitivity import build
 data=build();data[np.isclose(data.gain_scale,1)].to_csv(TABLES/'Figure2_selected_gain_operating_points.csv',index=False)
 f=fig(252);rows=f.add_gridspec(2,1,left=.090,right=.978,top=.974,bottom=.135,height_ratios=[1.26,2.05],hspace=.095)
 top=rows[0].subgridspec(1,3,width_ratios=[2.05,.65,1.32],wspace=.36)
 ax=f.add_subplot(top[0]);letter(ax,'A',x=-.13,y=1.005);memory_mechanism(ax)
 # Give the two-group objective display a compact axis; A/C keep their height.
 score_slot=top[1].subgridspec(2,1,height_ratios=[.60,.40],hspace=0)
 ax=f.add_subplot(score_slot[0]);letter(ax,'B',y=1.005);score_panel(ax);ax.tick_params(axis='x',labelsize=5.7)
 ax.set_xticklabels(['D\nperiod','Coupling'])
 for text in list(ax.texts):
  if text.get_text().startswith('n = '):text.remove()
 ax=f.add_subplot(score_slot[1]);ax.axis('off')
 nruns=len(list((DATA/'optimization').glob('seed_*/fit_D-period-X_main_protocol.json')))
 ax.text(.5,.66,f'n = {nruns} search seeds\nper architecture',ha='center',va='center',fontsize=6.3)
 ax.text(.5,.20,'Identical targets,\nshared bounds,\nobjective, budget\nand period estimator',ha='center',va='center',fontsize=6.3,linespacing=1.2)
 sub=top[2].subgridspec(3,1,hspace=.24);a=[f.add_subplot(sub[i]) for i in range(3)];letter(a[0],'C',x=-.31);comparison_rows(a)
 for ax in a[:2]:ax.tick_params(labelbottom=False)
 bottom=rows[1].subgridspec(1,2,wspace=.36)
 from matplotlib.ticker import FuncFormatter,MaxNLocator
 # Pair every scale and reference across the two architectures. In particular,
 # do not magnify the relatively small period-adaptive mismatch in its gain row.
 paired_axes=[]
 driver_max=float(max(abs(data.loc[data.architecture.eq('D-period-X'),'release_delta_omega_rad_h']).max(),abs(data.loc[data.architecture.eq('coupling-X'),'release_K_eff_rad_h']).max()))
 driver_limit=np.ceil(driver_max/.05)*.05
 period_values=data[['D_period_refined_h','V_period_refined_h']].to_numpy()
 period_limits=(float(np.floor(np.nanmin(period_values))),float(np.ceil(np.nanmax(period_values))))
 ratio_values=data.release_locking_ratio.to_numpy()
 finite_ratio=ratio_values[np.isfinite(ratio_values)&(ratio_values>0)]
 ratio_limits=(10.**np.floor(np.log10(finite_ratio.min())),10.**np.ceil(np.log10(finite_ratio.max())))
 shared_limits=[(-driver_limit,driver_limit),(-12.7,12.7),period_limits,ratio_limits]
 for i,arch in enumerate(['D-period-X','coupling-X']):
  sub=bottom[i].subgridspec(4,1,hspace=.39,height_ratios=[.95,1,1,.95])
  axes=[f.add_subplot(sub[j]) for j in range(4)];letter(axes[0],'DE'[i],x=-.18)
  paired_axes.append(axes)
  z=data[data.architecture==arch]
  driver='release_delta_omega_rad_h' if i==0 else 'release_K_eff_rad_h'
  driver_label=(r'$\Delta\omega_{0}$' if i==0 else r'$K_{{\rm eff},0}$')+r' ($10^{-3}$ rad h$^{-1}$)'
  sensitivity_lines(axes[0],z,'gain_scale',driver)
  axes[0].set(xlabel='D-period gain / selected gain' if i==0 else 'Coupling gain / selected gain',ylabel=driver_label,xticks=[0,.5,1,1.5,2])
  axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{1000*x:g}'))
  axes[0].yaxis.set_major_locator(MaxNLocator(5))
  axes[0].axhline(0,color='#888888',ls=':',lw=.6,zorder=0)
  sensitivity_lines(axes[1],z,driver,'gap_circular_7d_h',phase=True)
  axes[1].set_ylabel('Circular D–V gap (h)')
  axes[1].set_yticks([-12,-6,0,6,12])
  axes[1].axhline(0,color='#888888',ls=':',lw=.6,zorder=0)
  sensitivity_lines(axes[2],z,driver,'D_period_refined_h')
  sensitivity_lines(axes[2],z,driver,'V_period_refined_h',ls='--',selected=False)
  axes[2].set_ylabel('Regional DD period (h)')
  axes[2].set_yticks(np.arange(period_limits[0],period_limits[1]+.1,2))
  axes[2].axhline(24,color='#888888',ls=':',lw=.6,zorder=0)
  sensitivity_lines(axes[3],z,driver,'release_locking_ratio')
  axes[3].set(yscale='log',ylabel=r'Release $|\Delta\omega/K_{\rm eff}|$',xlabel=driver_label)
  axes[3].axhline(1,color='#555555',ls='--',lw=.65)
  low,high=z[driver].min(),z[driver].max();pad=(high-low)*.05
  for ax in axes[1:]:
   ax.set_xlim(low-pad,high+pad)
   ax.xaxis.set_major_locator(MaxNLocator(5))
   ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{1000*x:g}'))
   ax.axvline(0,color='#888888',ls=':',lw=.6,zorder=0)
  for ax in axes[1:3]:ax.tick_params(labelbottom=False)
  for ax,limits in zip(axes,shared_limits):ax.set_ylim(*limits)
  for ax in axes:ax.tick_params(labelsize=5.9);ax.yaxis.label.set_size(6.5);ax.xaxis.label.set_size(6.5)
 # Preserve an inspectable record of display equality and fitted provenance.
 selected={}
 for arch in ['D-period-X','coupling-X']:
  runs=[json.loads(p.read_text()) for p in sorted((DATA/'optimization').glob(f'seed_*/fit_{arch}_main_protocol.json'))]
  q=chosen(arch)
  selected[arch]={'selected_seed':q['seed'],'selected_final_objective':q['final_diagnostic_score'],'all_run_final_objectives':[r['final_diagnostic_score'] for r in runs],'selected_is_lowest_recorded_final_objective':q['final_diagnostic_score']==min(r['final_diagnostic_score'] for r in runs)}
 checks={'paired_y_limits':{name:[list(paired_axes[i][j].get_ylim()) for i in range(2)] for j,name in enumerate(['gain_driver_rad_h','circular_phase_h','regional_period_h','release_locking_ratio'])},'all_paired_y_limits_identical':all(np.array_equal(paired_axes[0][j].get_ylim(),paired_axes[1][j].get_ylim()) for j in range(4)),'matching_references':{'gain_driver':'horizontal dotted zero in both','circular_phase':'horizontal and vertical dotted zero in both','regional_period':'horizontal dotted 24 h and vertical dotted driver zero in both; D solid and V dashed','release_locking_ratio':'horizontal dashed one and vertical dotted driver zero in both'},'selected_fits':selected,'scope':'Best among three completed recorded search seeds per architecture, not a proof of a global optimum. Gain sweeps perturb the selected solution; only gain scale one is that selected fit.'}
 (ROOT/'validation/Figure2_display_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
 f.legend(handles=[Line2D([],[],marker='o',color='#555555',ls='',label='Period-adaptive'),Line2D([],[],marker='s',color='#555555',mfc='white',ls='',label='Coupling-adaptive'),Line2D([],[],marker='_',color='black',ls='',label='Calibration target')],loc='lower center',bbox_to_anchor=(.53,.07),ncol=3,frameon=False,fontsize=5.8)
 f.legend(handles=[Line2D([],[],color='#555555',ls='-',label='D period'),Line2D([],[],color='#555555',ls='--',label='V period'),Line2D([],[],marker='o',color='#777777',mec='black',ls='',label='Selected gain (D, E)')],loc='lower center',bbox_to_anchor=(.53,.041),ncol=3,frameon=False,fontsize=5.8)
 schedule_legend(figobj=f,bbox=(.53,.011));save(f,'Figure_2')

def s3():
 arches=['D-period-X','V-period-X','joint-DV-period-X'];qs=[chosen(a) for a in arches];target=plotting_targets()
 f=fig(131);gs=f.add_gridspec(2,3,left=.11,right=.98,top=.94,bottom=.13,wspace=.45,hspace=.46)
 ax=f.add_subplot(gs[0,0]);letter(ax,'A');ax.scatter(range(3),objective_display([q['final_diagnostic_score'] for q in qs]),c='#555555',s=24);ax.set(xticks=range(3),xticklabels=['D only','V only','D+V']);objective_axis(ax)
 axes=[f.add_subplot(gs[0,1]),f.add_subplot(gs[0,2]),f.add_subplot(gs[1,:2])]
 for i,q in enumerate(qs):
  z=pd.DataFrame(q['selected']['endpoints']);mark=['o','s','^'][i]
  for ax,(ex,col,tgt,se,ylab) in zip(axes,[(False,'behavior_h','behavior_mean_h','behavior_display_sem_h','Behavior (h)'),(True,'behavior_h','exvivo_scn_target_h',None,'Ex vivo SCN (h)'),(True,'gap_7day_folded_arithmetic_h','gap_mean_h','gap_sem_h','D–V gap (h)')]):
   for j,c in enumerate(ORDER):
    r=z[(z.condition==c)&z.ex_vivo.eq(ex)].iloc[0];ax.plot(j+(i-1)*.18,r[col],marker=mark,ls='',color=COL[c],ms=3.6,mfc=COL[c] if i==0 else 'white')
    if i==0:ax.errorbar(j,target.loc[c,tgt],yerr=None if se is None else target.loc[c,se],fmt='_',color='black',capsize=1.8,ms=6,lw=.5)
   conditions(ax,ylab)
 for ax,label in zip(axes,'BCD'):letter(ax,label,x=-.2 if label!='D' else -.09)
 ax=f.add_subplot(gs[1,2]);ax.axis('off');ax.legend(handles=[Line2D([],[],marker=m,ls='',color='#555555',label=n,mfc='#555555' if i==0 else 'white') for i,(m,n) in enumerate(zip(['o','s','^'],['D only','V only','D+V']))],frameon=False,loc='upper left')
 schedule_legend(figobj=f,bbox=(.53,.02));save(f,'Figure_S3',True)
if __name__=='__main__':main2();s3()
