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
     ax.plot([i+off,i+off],[r.D_h-baseline,r.V_h-baseline],color=COL[c],lw=1.2,alpha=.75)
     if residual:
      ax.plot(i+off-.025,r.D_h-baseline,marker=mark,ls='',ms=4,mfc=COL[c] if ai==0 else 'white',mec=COL[c])
      ax.plot(i+off+.025,r.V_h-baseline,marker=mark,ls='',ms=4,mfc=COL[c] if ai==0 else 'white',mec=COL[c])
    ax.plot(i+off,v,marker=mark,ls='',ms=4.3,mfc=COL[c] if ai==0 else 'white',mec=COL[c],mew=.95)
    if ai==0:
     ax.errorbar(i,0 if residual else target.loc[c,tkey],yerr=None if se is None else target.loc[c,se],fmt='_',ms=6,color='black',capsize=2,lw=1.0)
   conditions(ax,('Residual (h)' if residual else ylabel))



def sensitivity_lines(ax,data,xkey,ykey,ls='-',phase=False,selected=True):
 """Connect full forward simulations in gain order, never across a wrap cut."""
 for c in ORDER:
  z=data[data.condition==c].sort_values('gain_scale')
  x=z[xkey].to_numpy();y=z[ykey].to_numpy().copy()
  y[~np.isfinite(y)]=np.nan
  cuts=np.abs(np.diff(y))>12 if phase else np.zeros(len(y)-1,dtype=bool)
  if ykey=='release_locking_ratio':cuts|=z.release_K_eff_rad_h.to_numpy()[:-1]*z.release_K_eff_rad_h.to_numpy()[1:]<0
  for ids in np.split(np.arange(len(z)),np.flatnonzero(cuts)+1):
   ax.plot(x[ids],y[ids],color=COL[c],ls=ls,lw=1.35)
  if selected:
   r=z[np.isclose(z.gain_scale,1)].iloc[0]
   ax.plot(r[xkey],r[ykey],marker='o',ms=4.4,ls='',mfc=COL[c],mec='black',mew=.65,zorder=5)


def memory_mechanism(ax):
 """Dorsal-memory decomposition and the shared systemic period memory.

 Q_D and P_D start at zero and use the same linear filter, so their sum obeys
 the original M_D equation exactly. Chi gates the exposure interval, including
 its dark phases; it is zero only after release to DD. No separate states are
 added to the simulator by this explanatory diagram. The X card exposes its
 existing period memory, shared by both architectures; it does not specify an
 exclusively SCN-mediated acquisition route for the effective history drive.
 """
 from matplotlib.patches import Ellipse,Rectangle
 ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 # Reserve one warm-brown accent for mechanisms, outside the schedule palette.
 # Circle/square symbols distinguish architectures; node identity follows the
 # red/blue/charcoal activity traces used throughout the manuscript.
 period_color=coupling_color='#795438'
 def label(x,y,text,fontsize=11,**kwargs):
  return ax.text(x,y,text,ha='center',va='center',fontsize=fontsize,**kwargs)
 def arrow(a,b,color='#454545',width=1.25,style='-|>'):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,lw=width,color=color,shrinkA=1,shrinkB=1))
 label(.5,.990,'Light schedule  '+r'$I(t),\ \theta(t)$',fontsize=10.5,fontweight='bold')
 label(.5,.936,r'$\chi=1$ during entrainment; '+r'$\chi=0$ in DD',fontsize=9.2)
 for x,color in [(.015,'#F3EEE7'),(.535,'#F3EEE7')]:
  ax.add_patch(Rectangle((x,.660),.45,.235,facecolor=color,edgecolor='none'))
 label(.24,.862,'Photoperiod',fontsize=10.5,fontweight='bold')
 label(.76,.862,'Phase mismatch',fontsize=10.5,fontweight='bold')
 label(.24,.785,r'$u_P=g_{P,D}\left[\frac{1+\sin\phi_D}{2}-I\right]$',fontsize=11.5)
 label(.76,.785,r'$u_\phi=g_{\phi,D}\,I\,\sin(\theta-\phi_D)$',fontsize=11.5)
 label(.24,.703,r'$T_m\dot Q_D=\chi u_P-Q_D$')
 label(.76,.703,r'$T_m\dot P_D=\chi u_\phi-P_D$')
 arrow((.29,.658),(.445,.634));arrow((.71,.658),(.555,.634))
 label(.5,.600,r'$M_D=P_D+Q_D$',fontsize=11.5,fontweight='bold')
 arrow((.45,.573),(.25,.530),period_color);arrow((.55,.573),(.75,.530),coupling_color)
 label(.24,.500,'Adaptive period  ●',fontsize=10.5,color=period_color,fontweight='bold')
 label(.76,.500,'Adaptive coupling  □',fontsize=10.5,color=coupling_color,fontweight='bold')
 label(.24,.443,r'$\tau_D=\tau_{D,0}-g_D\,\tanh M_D$',fontsize=11.5)
 label(.76,.443,r'$K_{VD}=K_{VD,0}+g_K\,\tanh M_D$',fontsize=11.5)
 label(.24,.385,r'$K_{VD},\ K_{DV},\ \tau_V$ fixed',fontsize=9.2)
 label(.76,.385,r'$\tau_D,\ \tau_V,\ K_{DV}$ fixed',fontsize=9.2)
 # Convert physical diameters to axis coordinates: nodes remain true circles
 # even though the architecture occupies a wide, shallow panel.
 ax_mm=np.array([ax.get_position().width*ax.figure.get_figwidth()*25.4,
                 ax.get_position().height*ax.figure.get_figheight()*25.4])
 diameter_mm=10.5;radius_mm=diameter_mm/2
 xy={'D':(.32,.245),'V':(.68,.245),'X':(.50,.080)}
 for name,(x,y) in xy.items():
  ax.add_patch(Ellipse((x,y),*(diameter_mm/ax_mm),facecolor=REG[name],edgecolor=REG[name],lw=1.25,zorder=4))
  label(x,y,name,fontsize=13,color='white',fontweight='bold',zorder=5)
 def node_edge(name,toward):
  center=np.array(xy[name]);direction=(np.array(toward)-center)*ax_mm
  return center+(radius_mm+.35)*direction/np.linalg.norm(direction)/ax_mm
 # The adaptive K_VD term acts from D to V; the reverse K_DV stays fixed.
 dy=.025;dx=np.sqrt(radius_mm**2-(dy*ax_mm[1])**2)/ax_mm[0]+.003
 arrow((xy['D'][0]+dx,.245-dy),(xy['V'][0]-dx,.245-dy),coupling_color,1.55)
 arrow((xy['D'][0]+dx,.245+dy),(xy['V'][0]-dx,.245+dy),'#777777',style='<-')
 arrow((.24,.354),node_edge('D',(.24,.354)),period_color,1.4)
 arrow((.71,.355),(.54,.245-dy),coupling_color,1.4)
 for left,right in [('D','X'),('V','X')]:
  arrow(node_edge(left,xy[right]),node_edge(right,xy[left]),'#777777',1.15,'<->')
 # X has its own effective history-dependent period in both architectures.
 # Its implemented memory drive contains I and theta explicitly, so do not
 # imply that the reciprocal SCN-X phase connections are its sole input.
 ax.add_patch(Rectangle((.600,.003),.398,.155,facecolor='#F3EEE7',
                        edgecolor=period_color,lw=.95,zorder=1))
 label(.799,.135,'Systemic history',fontsize=9.2,
       color=period_color,fontweight='bold')
 label(.799,.084,r'$\tau_X=\tau_{X,0}-g_X\,\tanh M_X$',fontsize=11)
 label(.799,.031,'Shared by both architectures',fontsize=8.5,color=period_color)
 arrow((.598,.080),node_edge('X',(.598,.080)),period_color,1.4)


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
 height_mm=220.;f=fig(height_mm)
 # Preserve the enlarged architecture at its physical size. The quantitative
 # comparisons share four aligned columns; D and E each occupy one row.
 architecture_height_mm=73.77257811385363
 ax=f.add_axes([.105,(height_mm-6.5-architecture_height_mm)/height_mm,
                .875,architecture_height_mm/height_mm])
 letter(ax,'A',x=-.095,y=1.005);memory_mechanism(ax)
 # A has the full printable width; the objective and endpoint readouts share
 # one compact row. Symbols, fixed parameters and search design are captioned.
 endpoint_row=f.add_gridspec(1,4,left=.105,right=.98,
                            bottom=105.75/height_mm,top=130.75/height_mm,wspace=.52)
 ax=f.add_subplot(endpoint_row[0]);letter(ax,'B',x=-.50,y=1.08);score_panel(ax)
 ax.set_xticklabels(['D\nperiod','Coupling'])
 ax.set_ylabel('Penalized objective\n'+r'($\times 10^6$)')
 for text in list(ax.texts):
  if text.get_text().startswith('n = '):text.remove()
 for index,text in enumerate(t for t in ax.texts if hasattr(t,'xy')):
  text.set_position((8 if index==0 else -8,0))
  text.set(ha='left' if index==0 else 'right',va='center')
 a=[f.add_subplot(endpoint_row[i]) for i in range(1,4)];letter(a[0],'C',x=-.43,y=1.08);comparison_rows(a)
 for ax,title in zip(a,['Behavior (h)','Ex vivo SCN (h)','D–V gap (h)']):
  ax.set_ylabel('');ax.set_title(title,pad=5,fontsize=8.5)
  ax.tick_params(axis='x',labelrotation=45)
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
  bottom_mm,top_mm=[(60.,87.),(16.,43.)][i]
  sub=f.add_gridspec(1,4,left=.105,right=.98,bottom=bottom_mm/height_mm,
                     top=top_mm/height_mm,wspace=.52)
  axes=[f.add_subplot(sub[j]) for j in range(4)]
  f.text(.021,(top_mm+4)/height_mm,'DE'[i],fontsize=11,fontweight='bold',va='bottom')
  f.text(.105,(top_mm+4)/height_mm,['Adaptive period','Adaptive coupling'][i],
         fontsize=9.5,fontweight='bold',va='bottom')
  paired_axes.append(axes)
  z=data[data.architecture==arch]
  driver='release_delta_omega_rad_h' if i==0 else 'release_K_eff_rad_h'
  driver_label=(r'$\Delta\omega_{0}$' if i==0 else r'$K_{{\rm eff},0}$')+'\n'+r'($10^{-3}$ rad h$^{-1}$)'
  sensitivity_lines(axes[0],z,'gain_scale',driver)
  axes[0].set(xlabel='Gain / selected gain',ylabel=(r'$\Delta\omega_{0}$' if i==0 else r'$K_{{\rm eff},0}$')+'\n'+r'($10^{-3}$ rad h$^{-1}$)',xticks=[0,1,2])
  axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{1000*x:g}'))
  axes[0].yaxis.set_major_locator(MaxNLocator(5))
  axes[0].axhline(0,color='#999999',ls=':',lw=.85,zorder=0)
  sensitivity_lines(axes[1],z,driver,'gap_circular_7d_h',phase=True)
  axes[1].set_ylabel('Circular D–V\ngap (h)')
  axes[1].set_yticks([-12,-6,0,6,12])
  axes[1].axhline(0,color='#999999',ls=':',lw=.85,zorder=0)
  sensitivity_lines(axes[2],z,driver,'D_period_refined_h')
  sensitivity_lines(axes[2],z,driver,'V_period_refined_h',ls='--',selected=False)
  axes[2].set_ylabel('DD period (h)')
  axes[2].set_yticks(np.arange(period_limits[0],period_limits[1]+.1,2))
  axes[2].axhline(24,color='#999999',ls=':',lw=.85,zorder=0)
  sensitivity_lines(axes[3],z,driver,'release_locking_ratio')
  axes[3].set(yscale='log',ylabel='Release ratio\n'+r'$|\Delta\omega/K_{\rm eff}|$',xlabel=driver_label)
  axes[3].axhline(1,color='#777777',ls='--',lw=1)
  low,high=z[driver].min(),z[driver].max();pad=(high-low)*.05
  for ax in axes[1:]:
   ax.set_xlim(low-pad,high+pad)
   ax.set_xlabel(driver_label)
   ax.set_xticks([-.015,0,.015,.030] if i==0 else [-.1,0,.1,.2])
   ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{1000*x:g}'))
   ax.axvline(0,color='#999999',ls=':',lw=.85,zorder=0)
  for ax,limits in zip(axes,shared_limits):ax.set_ylim(*limits)
  for ax in axes:
   ax.tick_params(labelsize=8,pad=2)
   ax.yaxis.label.set_size(8);ax.yaxis.labelpad=3
   ax.xaxis.label.set_size(8);ax.xaxis.labelpad=3
 # Preserve an inspectable record of display equality and fitted provenance.
 selected={}
 for arch in ['D-period-X','coupling-X']:
  runs=[json.loads(p.read_text()) for p in sorted((DATA/'optimization').glob(f'seed_*/fit_{arch}_main_protocol.json'))]
  q=chosen(arch)
  selected[arch]={'selected_seed':q['seed'],'selected_final_objective':q['final_diagnostic_score'],'all_run_final_objectives':[r['final_diagnostic_score'] for r in runs],'selected_is_lowest_recorded_final_objective':q['final_diagnostic_score']==min(r['final_diagnostic_score'] for r in runs)}
 checks={'paired_y_limits':{name:[list(paired_axes[i][j].get_ylim()) for i in range(2)] for j,name in enumerate(['gain_driver_rad_h','circular_phase_h','regional_period_h','release_locking_ratio'])},'all_paired_y_limits_identical':all(np.array_equal(paired_axes[0][j].get_ylim(),paired_axes[1][j].get_ylim()) for j in range(4)),'matching_references':{'gain_driver':'horizontal dotted zero in both','circular_phase':'horizontal and vertical dotted zero in both','regional_period':'horizontal dotted 24 h and vertical dotted driver zero in both; D solid and V dashed','release_locking_ratio':'horizontal dashed one and vertical dotted driver zero in both'},'selected_fits':selected,'scope':'Best among three completed recorded search seeds per architecture, not a proof of a global optimum. Gain sweeps perturb the selected solution; only gain scale one is that selected fit.'}
 (ROOT/'validation/Figure2_display_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
 save(f,'Figure_2')

def s3():
 arches=['D-period-X','V-period-X','joint-DV-period-X'];qs=[chosen(a) for a in arches];target=plotting_targets()
 f=fig(113);gs=f.add_gridspec(2,2,left=.115,right=.98,top=.94,bottom=.10,wspace=.38,hspace=.45)
 ax=f.add_subplot(gs[0,0]);letter(ax,'A');ax.scatter(range(3),objective_display([q['final_diagnostic_score'] for q in qs]),c='#555555',s=28);ax.set(xticks=range(3),xticklabels=['D only','V only','D+V']);objective_axis(ax)
 axes=[f.add_subplot(gs[0,1]),f.add_subplot(gs[1,0]),f.add_subplot(gs[1,1])]
 for i,q in enumerate(qs):
  z=pd.DataFrame(q['selected']['endpoints']);mark=['o','s','^'][i]
  for ax,(ex,col,tgt,se,ylab) in zip(axes,[(False,'behavior_h','behavior_mean_h','behavior_display_sem_h','Behavior (h)'),(True,'behavior_h','exvivo_scn_target_h',None,'Ex vivo SCN (h)'),(True,'gap_7day_folded_arithmetic_h','gap_mean_h','gap_sem_h','D–V gap (h)')]):
   for j,c in enumerate(ORDER):
    r=z[(z.condition==c)&z.ex_vivo.eq(ex)].iloc[0];ax.plot(j+(i-1)*.18,r[col],marker=mark,ls='',color=COL[c],ms=4.5,mfc=COL[c] if i==0 else 'white',mew=1)
    if i==0:ax.errorbar(j,target.loc[c,tgt],yerr=None if se is None else target.loc[c,se],fmt='_',color='black',capsize=2,ms=6,lw=1)
   conditions(ax,ylab)
 for ax,label in zip(axes,'BCD'):letter(ax,label,x=-.2)
 save(f,'Figure_S3',True)
if __name__=='__main__':main2();s3()
