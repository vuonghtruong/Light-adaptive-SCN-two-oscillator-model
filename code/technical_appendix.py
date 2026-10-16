from common import *
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

def main():
 (ROOT/'work').mkdir(parents=True,exist_ok=True)
 out=ROOT/'figures/supplement/Technical_appendix.pdf'
 page_index=1
 with PdfPages(out) as book:
  f=fig(94);gs=f.add_gridspec(1,2,left=.10,right=.98,top=.93,bottom=.30,wspace=.40,width_ratios=[1.25,1])
  ax=f.add_subplot(gs[0,0]);letter(ax,'A',x=-.13);z=read('reference_long_DD.csv')
  for c in ORDER:
   q=z[z.condition==c];ax.plot((q.window_start_day+q.window_end_day)/2,q.behavior_h,color=COL[c])
  ax.set(xlabel='DD time (days)',ylabel='Behavioral period (h)')
  ax=f.add_subplot(gs[0,1]);letter(ax,'B',x=-.18);z=read('perturbation_memory_contrasts.csv');tests=['reference','D_memory_off','X_memory_off','D_and_X_memory_off']
  for c,mark in [('T22','o'),('T26','s')]:
   q=z[(z.condition==c)&(~z.ex_vivo)].set_index('test');ax.plot(range(4),[q.loc[t,'behavior_h_minus_LD_h'] for t in tests],color=COL[c],marker=mark,ms=3)
  ax.axhline(0,color='#777777',lw=.5)
  ax.set(xticks=range(4),xticklabels=['Retained','D off','X off','Both off'],ylabel='Behavioral period difference\nfrom matched LD (h)',xlabel='Memory adaptation');ax.tick_params(axis='x',labelrotation=30,labelsize=6)
  schedule_legend(figobj=f,bbox=(.53,.095));f.text(.10,.045,'Technical appendix, page 1. Long-DD relaxation (5-day windows) and memory controls.',fontsize=6.6)
  prepare_figure(f);book.savefig(f);f.savefig(ROOT/'work'/f'appendix_page_{page_index}.png',dpi=110);page_index+=1;plt.close(f)
  # Local diagnostics around the selected adaptive-D parameters.
  f=fig(102);gs=f.add_gridspec(1,3,left=.10,right=.98,top=.93,bottom=.28,wspace=.63)
  ax=f.add_subplot(gs[0,0]);letter(ax,'A');z=read('original_parameter_landscape.csv').pivot(index='gp_d',columns='gphi_d',values='weighted_sse')/OBJECTIVE_DISPLAY_SCALE;im=ax.pcolormesh(z.columns,z.index,z,norm=LogNorm(z.min().min(),np.percentile(z,98)),cmap='viridis',shading='auto',rasterized=True);cax=f.add_axes([.10,.15,.22,.014]);cb=f.colorbar(im,cax=cax,orientation='horizontal',format='%g');cb.locator=LogLocator(base=10,subs=[1,2,5]);cb.update_ticks();cb.minorticks_off();cb.ax.tick_params(labelsize=6.3);cb.set_label(OBJECTIVE_LABEL,labelpad=1,fontsize=6.3);ax.set(xlabel=r'$g_{\phi,D}$',ylabel=r'$g_{P,D}$')
  ax=f.add_subplot(gs[0,1]);letter(ax,'B');z=read('reference_frozen_phase_flow.csv')
  for c in ORDER:
   q=z[z.condition==c];ax.plot(q.delta_rad,q.phase_gap_derivative_rad_h,color=COL[c])
  ax.axhline(0,color='#777777',lw=.5);ax.set(xlabel='D–V phase (rad)',ylabel='Phase velocity (rad/h)')
  ax=f.add_subplot(gs[0,2]);letter(ax,'C');q=read('full_model_dynamic_traces.csv.gz');p=__import__('reference_model').PARAMS
  for c in ['T22','T26']:
   z=q[(q.condition==c)&q.context.eq('invivo')&(q.time_from_DD_h>=0)&(q.time_from_DD_h<=168)];ax.plot(z.time_from_DD_h/24,z.tau_D,color=COL[c]);ax.plot(z.time_from_DD_h/24,z.tau_X,color=COL[c],ls='--')
  ax.axhline(p['tau_v'],color='#777777',ls=':',lw=.7);ax.set(xlabel='DD day',ylabel='Natural period (h)')
  ax.legend(handles=[Line2D([],[],color='#555555',label='D'),Line2D([],[],color='#555555',ls='--',label='X'),Line2D([],[],color='#777777',ls=':',label='V')],frameon=False,fontsize=6.3,ncol=3,handlelength=.9,columnspacing=.45,loc='upper right')
  schedule_legend(figobj=f,bbox=(.66,.075));f.text(.10,.022,'Technical appendix, page 2. Local penalized-objective landscape, phase flow and natural periods.',fontsize=6.6)
  prepare_figure(f);book.savefig(f);f.savefig(ROOT/'work'/f'appendix_page_{page_index}.png',dpi=110);page_index+=1;plt.close(f)
  f=fig(190);axs=f.subplots(3,3);f.subplots_adjust(left=.10,right=.98,top=.96,bottom=.13,wspace=.41,hspace=.50);b=read('original_logic_bifurcation_branches.csv');order=['LD','SP','LP','18:6 LD','T22','T23','T25','T26']
  for i,c in enumerate(order):
   ax=axs.flat[i];z=b[b.condition==c]
   for stability,ls in [('stable','-'),('unstable','--')]:
    q=z[z.stability==stability].sort_values('K_VD_rad_h')
    effective=(q.K_VD_rad_h-q.K_DV_rad_h).to_numpy()
    cuts=np.flatnonzero((effective[:-1]*effective[1:]<=0)|(np.abs(np.diff(q.phase_rad))>np.pi))+1
    for indices in np.split(np.arange(len(q)),cuts):
     segment=q.iloc[indices];ax.plot(segment.K_VD_rad_h,segment.phase_rad,color=COL[c],ls=ls)
   letter(ax,chr(65+i));ax.set(xlabel=r'$K_{VD}$ (rad/h)',ylabel='Phase (rad)',yticks=[-np.pi,0,np.pi],yticklabels=['−π','0','π']);ax.text(.97,.94,c.replace(' LD',''),transform=ax.transAxes,ha='right',va='top',fontsize=6.3,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.5))
  ax=axs.flat[8];letter(ax,'I');z=read('original_logic_phase_reorganization.csv')
  for c in ['LD','18:6 LD','T23','T25']:
   q=z[z.condition==c];ax.plot(q.DD_h/24,q.instantaneous_frozen_ratio,color=COL[c])
  ax.axhline(1,color='black',ls='--',lw=.7);ax.set(xlabel='DD day',ylabel=r'$|\Delta\omega(t)/K_{\rm eff}|$')
  f.text(.10,.050,'Technical appendix, page 3. Frozen bifurcation curves (solid stable; dashed unstable)',fontsize=7)
  f.text(.10,.029,'and time-varying locking diagnostic; neither proves locking with decaying memory.',fontsize=6.3)
  prepare_figure(f);book.savefig(f);f.savefig(ROOT/'work'/f'appendix_page_{page_index}.png',dpi=110);page_index+=1;plt.close(f)
 print('Saved Technical_appendix.pdf',flush=True)
if __name__=='__main__':main()
