from pathlib import Path
import json,re,os
# Fixed export metadata and SVG IDs make repeated vector exports comparable.
os.environ.setdefault('SOURCE_DATE_EPOCH','1790121600')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,Circle,FancyArrowPatch
from matplotlib.text import Text
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';TABLES=ROOT/'tables'
ORDER=['SP','LP','LD','T22','T26']
COL={'SP':'#1B9E77','LP':'#7B3294','LD':'#4D4D4D','T22':'#E6AB02','T26':'#A6761D','18:6 LD':'#A50F15','T23':'#5E81AC','T25':'#C06C84'}
REG={'D':'#D73027','V':'#2C7BB6','X':'#222222'}
AGE={'young':'#36AA6F','adult':'#5B8DB8','old':'#C23B55','unknown':'#666666'}
REP={'Bmal1-ELuc':'#1B9E77','PER2::LUC':'#E67E22','Per1::GFP':'#8073AC','Per1-GFP':'#8073AC','Per1-luc':'#4C78A8'}
MARKERS=['o','s','^','D','v','P','X','<','>','h','*','p','d','8']
# Presentation only: never feed these scaled values back into fitting/selection.
OBJECTIVE_DISPLAY_SCALE=1e6
OBJECTIVE_LABEL='Penalized objective (×10⁶)'
def objective_display(raw):return np.asarray(raw)/OBJECTIVE_DISPLAY_SCALE
def objective_axis(ax):
 from matplotlib.ticker import FuncFormatter,LogLocator,NullFormatter
 ax.set(yscale='log',ylabel=OBJECTIVE_LABEL)
 ax.yaxis.set_major_locator(LogLocator(base=10,subs=[1,2,5]))
 ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
 ax.yaxis.set_minor_formatter(NullFormatter())
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.2,'axes.linewidth':.65,'lines.linewidth':.9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'svg.fonttype':'none','svg.hashsalt':'JBR-26-0241-v3','pdf.fonttype':42})
def read(name):return pd.read_csv(DATA/name)
def plotting_targets():
 """Add published behavioral SEMs for display without altering fitting inputs."""
 target=read('archived_targets.csv').set_index('condition').copy()
 observations=read('tcycle_period_observations.csv')
 # LD uses the Myung 2012 calibration mean, not the Azzi T24 observation.
 sources={'SP':('Myung et al.',2015,'8:16 LD'),
          'LP':('Myung et al.',2015,'16:8 LD'),
          'LD':('Myung et al.',2012,'12:12 LD'),
          'T22':('Azzi et al.',2017,'T22'),
          'T26':('Azzi et al.',2017,'T26')}
 for condition,(study,year,schedule) in sources.items():
  rows=observations[observations.study.eq(study)&observations.year.eq(year)&
                    observations.measurement.eq('Behavior')&
                    observations.condition_label.eq(schedule)]
  # Repeated reporter entries can describe the same behavioral measurement.
  values=rows[['mean_period_h','sem_h']].drop_duplicates()
  if len(values)!=1:
   raise ValueError(f'Ambiguous or missing behavioral SEM source for {condition}')
  mean,sem=values.iloc[0].to_numpy(dtype=float)
  if not np.isfinite(sem) or sem<0 or not np.isclose(mean,target.loc[condition,'behavior_mean_h'],rtol=0,atol=1e-12):
   raise ValueError(f'Behavioral SEM source does not match target for {condition}')
  target.loc[condition,'behavior_display_sem_h']=sem
  target.loc[condition,'behavior_display_source']=f'{study} ({year}), {schedule}'
 return target
def fig(height):return plt.figure(figsize=(175/25.4,height/25.4))
def letter(ax,label,x=-.2,y=1.04):ax.text(x,y,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
def prepare_figure(f):
 """Enlarge every text element once, including explicit small annotations."""
 f.canvas.draw()
 for artist in f.findobj(match=Text):
  if not getattr(artist,'_revision_font_scaled',False):
   artist.set_fontsize(max(7.2,artist.get_fontsize()*1.15))
   artist._revision_font_scaled=True
 f.canvas.draw()

def save(f,name,supp=False):
 prepare_figure(f)
 out=ROOT/'figures'/('supplement' if supp else 'main');out.mkdir(exist_ok=True,parents=True)
 for ext in ['pdf','svg','png']:f.savefig(out/(name+'.'+ext),dpi=230,facecolor='white')
 plt.close(f);print('Saved',name,flush=True)
def schedule_legend(ax=None,figobj=None,ncol=5,loc='lower center',bbox=(.53,.01),order=ORDER):
 handles=[Line2D([],[],color=COL[c],label=c.replace(' LD','')) for c in order]
 if figobj is not None:return figobj.legend(handles=handles,loc=loc,bbox_to_anchor=bbox,ncol=ncol,frameon=False)
 return ax.legend(handles=handles,loc=loc,ncol=ncol,frameon=False)
def conditions(ax,ylabel=None):
 ax.set_xticks(range(5),ORDER);ax.set_xlim(-.5,4.5)
 if ylabel:ax.set_ylabel(ylabel)
def chosen(arch):
 from parameter_registry import selected
 return selected(arch)
def selected_endpoints(arch):return pd.DataFrame(chosen(arch)['selected']['endpoints'])
def score_panel(ax):
 for i,arch in enumerate(['D-period-X','coupling-X']):
  qs=[json.loads(f.read_text()) for f in sorted((DATA/'optimization').glob(f'seed_*/fit_{arch}_main_protocol.json'))]
  vals=objective_display([q['final_diagnostic_score'] for q in qs])
  ax.scatter(i+np.linspace(-.09,.09,len(vals)),vals,marker=['o','s'][i],s=23,facecolors='#555555' if i==0 else 'white',edgecolors='#555555',alpha=.55)
  ax.scatter(i,min(vals),marker='*',s=65,color='black',zorder=5)
  ax.annotate(f'{min(vals):.3f}',(i,min(vals)),xytext=(0,-13),textcoords='offset points',fontsize=6.5,ha='center')
 ax.set(xticks=[0,1],xticklabels=['D period','Coupling'],xlim=(-.4,1.5))
 objective_axis(ax)
 ax.text(.5,-.18,f'n = {len(qs)} search seeds\nper architecture',transform=ax.transAxes,ha='center',va='top',fontsize=5.6)
 ax.margins(y=.25)
