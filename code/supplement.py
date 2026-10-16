from common import *
from matplotlib.colors import ListedColormap,BoundaryNorm
from scipy.signal import find_peaks

def s1():
 tr=read('reduction_traces.csv.gz');tab=read('reduction_targets.csv');targets=plotting_targets();records=json.loads((DATA/'reduction_parameters.json').read_text());primary=next(p['fit_id'] for p in records if p['architecture']=='D-period-X')
 f=fig(190);gs=f.add_gridspec(3,3,left=.105,right=.98,top=.955,bottom=.075,wspace=.50,hspace=.57);axes=[f.add_subplot(gs[i//3,i%3]) for i in range(8)]
 a=np.linspace(-np.pi/2,np.pi/2,400);axes[0].plot(a*24/(2*np.pi),np.cos(a),color=COL['SP'],label=r'$\cos\alpha$ (coupling)');axes[0].plot(a*24/(2*np.pi),np.sin(a),color=COL['LP'],label=r'$\sin\alpha$ (frequency)');axes[0].set(xlabel='X–SCN lag\n(phase-equivalent h)',ylabel='Normalized contribution');axes[0].legend(frameon=False,fontsize=5.4,loc='lower right')
 lag_values=[]
 for j,p in enumerate(records):
  vals=[]
  for i,c in enumerate(ORDER):
   g=tr[tr.fit_id.eq(p['fit_id'])&tr.condition.eq(c)&tr.time_from_DD_h.ge(0)];rates=g.loc[g.contraction>0,'contraction'];tau=1/rates.min();vals.append(tau);lag_values.append(tau);axes[1].scatter(tau,2-j+(i-2)*.045,color=COL[c],marker='o' if j==0 else 's',s=14,alpha=.8,edgecolors='none')
  axes[1].text(.98,.84-j*.32,f'{min(vals):.2f}–{max(vals):.2f} h',transform=axes[1].transAxes,ha='right',fontsize=6)
 axes[1].scatter(240,0,marker='D',color='#333333',s=27);axes[1].set(xscale='log',xlim=(min(lag_values)*.6,600),ylim=(-.5,2.5),yticks=[0,1,2],yticklabels=['Memory','Coupling fit','Period fit'],xlabel='Relaxation time (h)');axes[1].text(240,.2,'240 h',ha='center',fontsize=6)
 for c in ORDER:
  g=tr[tr.fit_id.eq(primary)&tr.condition.eq(c)].sort_values('time_from_DD_h')
  axes[2].plot(g.time_from_DD_h/24,g.lag*24/(2*np.pi),color=COL[c],lw=.6);axes[3].plot(g.time_from_DD_h/24,g.memory_X,color=COL[c],lw=.7)
 for ax in axes[2:4]:ax.axvline(0,color='black',ls='--',lw=.6);ax.set(xlim=(-30,14),xticks=[-30,0,14],xlabel='Time relative to DD\nrelease (days)')
 axes[2].set_ylabel('Predicted X–SCN lag\n(phase-equivalent h)');axes[3].set_ylabel(r'Systemic memory, $M_X$ (a.u.)')
 for ax,(ctx,col,tgt,sem,ylab) in zip(axes[4:],[('invivo','behavior_h','behavior_mean_h','behavior_display_sem_h','Behavioral period\nin vivo (h)'),('exvivo','D_h','exvivo_scn_target_h',None,'D period ex vivo (h)'),('exvivo','V_h','exvivo_scn_target_h',None,'V period ex vivo (h)'),('exvivo','gap_7day_h','gap_mean_h','gap_sem_h','D–V phase gap\nfirst 7 days (h)')]):
  for i,c in enumerate(ORDER):
   for variant,dx,filled in [('oscillatory_X',-.1,False),('adiabatic_coherent',.1,True)]:
    z=tab[(tab.fit_id==primary)&(tab.variant==variant)&(tab.condition==c)&(tab.context==ctx)].iloc[0];ax.plot(i+dx,z[col],'o',ms=3.7,mfc=COL[c] if filled else 'white',mec=COL[c],mew=.85)
   ax.errorbar(i,targets.loc[c,tgt],yerr=None if sem is None else targets.loc[c,sem],fmt='_',ms=7,color='black',capsize=2,lw=.6)
  conditions(ax,ylab)
 for i,ax in enumerate(axes):letter(ax,chr(65+i))
 ax=f.add_subplot(gs[2,2]);ax.axis('off');ax.legend(handles=[Line2D([],[],color=COL[c],label=c) for c in ORDER],loc='upper left',frameon=False,ncol=2)
 ax.add_artist(ax.get_legend());ax.legend(handles=[Line2D([],[],marker='o',color='#555555',mfc='white',ls='',label='Oscillatory X'),Line2D([],[],marker='o',color='#555555',ls='',label='Independent reduction'),Line2D([],[],marker='_',color='black',ls='',label='Fitting target')],loc='lower left',frameon=False,fontsize=5.6)
 save(f,'Figure_S1',True)

def s2():
 # Main Figure 3 contains the combined coupling/memory controls. Here isolate
 # readout weighting, a phase-only release perturbation, and feedback weighting.
 w=read('perturbation_weighting.csv');mem=read('explant_controls.csv')
 f=fig(90);axs=f.subplots(1,3);f.subplots_adjust(left=.095,right=.98,top=.92,bottom=.30,wspace=.53)
 for c in ORDER:
  z=w[(w.condition==c)&w.branch.eq('readout_only')].sort_values('D_weight');axs[0].plot(z.D_weight,z.behavior_h,'o-',ms=2.5,color=COL[c])
 axs[0].set(xlabel='D:V behavioral\nweighting',ylabel='Behavioral period (h)')
 phase=mem[np.isclose(mem.memory_D_retained,1)&np.isclose(mem.coupling_scale,1)&np.isclose(mem.X_feedback_retained,0)]
 for c in ['LD','T22','T26']:
  for key,mark in [('D_h','o'),('V_h','^')]:
   z=phase[phase.condition==c].sort_values('D_phase_advance_h');axs[1].plot(z.D_phase_advance_h,z[key],color=COL[c],marker=mark,ms=3,ls='-' if key=='D_h' else '--')
 axs[1].set(xlabel='D phase advance\nat explant (h)',ylabel='Ex vivo regional period (h)',xticks=[-2,0,2])
 for c in ORDER:
  z=w[w.condition==c].pivot(index='D_weight',columns='branch',values='behavior_h').sort_index()
  axs[2].plot(z.index,z.readout_and_feedback-z.readout_only,'o-',ms=2.5,color=COL[c])
 axs[2].axhline(0,color='#777777',lw=.5);axs[2].set(xlabel='D:V readout / feedback\nweighting',ylabel='Added feedback-weight\neffect (h)')
 for ax in [axs[0],axs[2]]:
  ax.set(xticks=[.5,.7,.8,.9,1],xticklabels=['50:50','70:30','80:20','90:10','100:0']);ax.tick_params(axis='x',labelrotation=60,labelsize=5.4)
 for i,a in enumerate(axs):letter(a,'ABC'[i])
 schedule_legend(figobj=f,bbox=(.5,.006));f.legend(handles=[Line2D([],[],marker='o',color='#555555',ls='-',label='D (B)'),Line2D([],[],marker='^',color='#555555',ls='--',label='V (B)')],loc='lower center',bbox_to_anchor=(.5,.072),ncol=2,frameon=False)
 save(f,'Figure_S2',True)

def s4():
 obs=pd.read_csv(TABLES/'Table_S1_literature_provenance.csv');obs['study_key']=obs.study+' '+obs.citation_year.astype(str);studies=sorted(obs.study_key.unique());conds=sorted(obs.condition_label.unique(),key=lambda x:(0 if 'LD' in x else 2 if x.startswith(('SP ','LP ')) else 1,[float(n) for n in re.findall(r'\d+(?:\.\d+)?',x)],x));codes={'behavior':'B','bulk':'S','D-SCN':'D','V-SCN':'V','A-SCN':'A','P-SCN':'P','VM-SCN':'m','VL-SCN':'l'}
 f=fig(184);gs=f.add_gridspec(2,3,left=.29,right=.985,top=.96,bottom=.13,hspace=.34,wspace=.36,height_ratios=[2.0,1]);ax=f.add_subplot(gs[0,:]);letter(ax,'A',x=-.38)
 for i,s in enumerate(studies):
  for j,c in enumerate(conds):
   g=obs[(obs.study_key==s)&(obs.condition_label==c)]
   if len(g):
    ax.add_patch(plt.Rectangle((j-.48,i-.48),.96,.96,color='#E9EEEB',lw=0));letters=''.join(codes[k] for k in codes if k in set(g.scn_region));letters=letters if len(letters)<=2 else letters[:2]+'\n'+letters[2:4]+('\n'+letters[4:] if len(letters)>4 else '');ax.text(j,i,letters,ha='center',va='center',fontsize=6.3,linespacing=.85)
 ax.set(xlim=(-.5,len(conds)-.5),ylim=(len(studies)-.5,-.5),xticks=range(len(conds)),xticklabels=[c.replace(' LD','') for c in conds],yticks=range(len(studies)),yticklabels=[s.replace(' et al.','')+'\n'+', '.join(obs.loc[obs.study_key==s,'reporter'].dropna().unique()) for s in studies]);ax.tick_params(axis='x',labelsize=6.3,labelrotation=65);ax.tick_params(axis='y',labelsize=6.3,length=0)
 ax=f.add_subplot(gs[1,:]);letter(ax,'B',x=-.38);z=obs[obs.study.str.contains('Azzi')&obs.measurement.eq('SCN')]
 for i,c in enumerate(['T22','T24','T26']):
  for j,region in enumerate(['bulk','D-SCN','V-SCN']):
   g=z[(z.condition_label==c)&(z.scn_region==region)].iloc[0];co=COL['LD'] if c=='T24' else COL[c]
   ax.errorbar(i+(j-1)*.14,g.mean_period_h,yerr=g.sem_h,fmt=['o','^','s'][j],ms=4,color=co,capsize=2,lw=.7)
 ax.set(xticks=range(3),xticklabels=['T22','T24','T26'],ylabel='Azzi: ex vivo period (h)')
 ax.legend(handles=[Line2D([],[],marker=m,ls='',color='#555555',label=l) for m,l in [('o','Whole SCN'),('^','Dorsal'),('s','Ventral')]],frameon=False,fontsize=6.3,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),columnspacing=1.0,handletextpad=.45)
 f.text(.055,.073,'B behavior · S whole SCN · D dorsal · V ventral · A anterior · P posterior',fontsize=6.3)
 f.text(.055,.049,'m/l ventromedial/ventrolateral. Reporter names appear below each study.',fontsize=6.3)
 f.text(.055,.025,'Table S7 lists regional means, reporters, ages and sample-size notes.',fontsize=6.3)
 save(f,'Figure_S4',True)
if __name__=='__main__':s1();s2();s4()
