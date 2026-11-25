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
from matplotlib.collections import LineCollection
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';TABLES=ROOT/'tables'
ORDER=['SP','LP','LD','T22','T26']
# One condition palette for every figure; age/reporter/region keys are separate.
# Core colors follow the distinguishable Okabe-Ito palette, with neutral LD.
COL={'SP':'#0072B2','LP':'#D55E00','LD':'#525252','T22':'#009E73','T26':'#CC79A7',
     '18:6 LD':'#7B3294','T23':'#56B4E9','T25':'#E69F00'}
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
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
 'legend.fontsize':8,'axes.linewidth':.95,'lines.linewidth':1.2,
 'xtick.major.width':.9,'ytick.major.width':.9,'xtick.major.size':3,'ytick.major.size':3,
 'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
 'svg.fonttype':'none','svg.hashsalt':'JBR-26-0241-publication','pdf.fonttype':42})
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
def letter(ax,label,x=-.2,y=1.04):ax.text(x,y,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
def prepare_figure(f):
 """Apply physical print sizes without changing plotted data or axis limits.

 Dense actograms retain narrower strokes/peak dots to resolve adjacent days;
 their renderer sets explicit publication sizes. Ensemble scatter stays intact.
 """
 f.canvas.draw()
 for artist in f.findobj(match=Text):
  if not getattr(artist,'_revision_font_scaled',False):
   artist.set_fontsize(max(8.,artist.get_fontsize()))
   artist._revision_font_scaled=True
 for ax in f.axes:
  for spine in ax.spines.values():spine.set_linewidth(.95)
  ax.tick_params(axis='both',which='both',width=.9)
 for line in f.findobj(match=Line2D):
  if getattr(line,'_publication_dense',False):continue
  if line.get_linestyle() not in ['None','none','',' ']:
   color=matplotlib.colors.to_rgb(line.get_color())
   neutral=max(color)-min(color)<.04
   minimum=.9 if neutral and line.get_linestyle() in [':','--'] else 1.2
   line.set_linewidth(max(minimum,line.get_linewidth()))
  if line.get_marker() not in ['None','none','',' ']:
   # Tick artists use marker codes; preserve their lengths.
   if isinstance(line.get_marker(),str):
    line.set_markersize(max(4.2,line.get_markersize()))
    line.set_markeredgewidth(max(.8,line.get_markeredgewidth()))
 for lines in f.findobj(match=LineCollection):
  widths=lines.get_linewidths()
  if len(widths):lines.set_linewidths(np.maximum(widths,.95))
 f.canvas.draw()

def save(f,name,supp=False):
 prepare_figure(f)
 out=ROOT/'figures'/('supplement' if supp else 'main');out.mkdir(exist_ok=True,parents=True)
 for ext in ['pdf','svg','png']:f.savefig(out/(name+'.'+ext),dpi=600,facecolor='white')
 record_publication_style(f,name)
 plt.close(f);print('Saved',name,flush=True)

def record_publication_style(f,name):
 """Record measurable export properties for review at the printed dimensions."""
 f.canvas.draw();renderer=f.canvas.get_renderer();bounds=f.bbox
 # Matplotlib keeps off-range tick labels as visible artists but does not draw
 # them. Exclude those from the physical-page clipping audit.
 undrawn_ticks=set()
 for ax in f.axes:
  for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
   lo,hi=sorted(limits);tol=max(1.,abs(lo),abs(hi))*1e-10
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():
    if not lo-tol<=tick.get_loc()<=hi+tol:
     undrawn_ticks.update([tick.label1,tick.label2])
 text=[a for a in f.findobj(match=Text) if a.get_visible() and a.get_text().strip() and a not in undrawn_ticks]
 outside=[]
 for a in text:
  box=a.get_window_extent(renderer)
  if box.x0<bounds.x0-1 or box.y0<bounds.y0-1 or box.x1>bounds.x1+1 or box.y1>bounds.y1+1:
   outside.append(a.get_text())
 report_path=ROOT/'validation/publication_style_checks.json'
 report=json.loads(report_path.read_text()) if report_path.exists() else {}
 report[name]=dict(width_mm=float(f.get_figwidth()*25.4),height_mm=float(f.get_figheight()*25.4),
                   minimum_text_pt=float(min(a.get_fontsize() for a in text)),
                   axes_linewidth_pt=.95,primary_linewidth_floor_pt=1.2,
                   dense_actogram_linewidth_pt=.70,PNG_dpi=600,
                   text_outside_canvas=outside,condition_palette=COL,
                   axes_bounds=[list(ax.get_position().bounds) for ax in f.axes])
 report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
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
