"""One renderer for all full-model double plots in this package.

Transplanted from the corrected original-manuscript plotting loop in
response_revision_2026-09-19/scripts/plot_main_figures.py::reference_figure2
and its actogram_style.py::positive_lobe. The v3 presentation removes DD
background shading, labels entrainment S1-S14, and stops at DD7.
"""
import numpy as np
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from common import REG
from reference_model import SCHEDULES

def positive_lobe(ax,x,values,baseline,height,color,*,connected=None,
                  fill_alpha=.23,linewidth=.5,linestyle='-',line_alpha=1.):
    x=np.asarray(x,dtype=float);values=np.asarray(values,dtype=float)
    visible=np.isfinite(values)&(values>0)
    if connected is not None:visible&=np.asarray(connected,dtype=bool)
    outline=np.where(visible,baseline+height*values,np.nan)
    if fill_alpha:
        ax.fill_between(x,baseline,baseline+height*values,where=visible,
                        interpolate=True,color=color,alpha=fill_alpha,lw=0)
    line,=ax.plot(x,outline,color=color,lw=linewidth,ls=linestyle,alpha=line_alpha)
    return line

def doubleplot_axis(ax,traces,condition,ex_vivo,*,show_ylabels=True,
                    show_xlabel=True,show_condition=True):
    a=traces[traces.condition.eq(condition)&traces.context.eq('exvivo' if ex_vivo else 'invivo')].sort_values('time_from_DD_h')
    if a.empty:raise ValueError('Missing full-model trajectory: '+condition)
    if 'model' in a and not a.model.eq('full_original_D_V_X').all():
        raise ValueError('Dynamic main panels require the full original model')
    t=a.time_from_DD_h.to_numpy();dt=float(np.median(np.diff(t)))
    peakt=t[find_peaks(a.behavior.to_numpy(),distance=max(1,int(round(16/dt))))[0]]
    cycle,photoperiod=SCHEDULES[condition]
    for row,start in enumerate(np.arange(-14,7)*24):
        # Do not expose DD8 in the right half of the final DD7 strip.
        s=a[(t>=start)&(t<=min(start+48,168))]
        xx=s.time_from_DD_h.to_numpy()-start;yy=20-row
        light=((s.time_from_DD_h<0)&(((s.time_from_DD_h+720)%cycle)<photoperiod)).astype(float)
        isdd=s.time_from_DD_h.to_numpy()>=0
        ax.fill_between(xx,yy,yy+.7*light,color='#F6C85F',alpha=.35,lw=0)
        connected=(~isdd) if ex_vivo else np.ones(len(s),dtype=bool)
        positive_lobe(ax,xx,s.X,yy,.72,REG['X'],connected=connected,
                      fill_alpha=0,linewidth=.42,linestyle=':',line_alpha=.75)
        ax.fill_between(xx,yy,yy+.72*np.maximum(s.behavior,0),where=connected,
                        color='#666666',alpha=.20,lw=0)
        for key in ['D','V']:positive_lobe(ax,xx,s[key],yy,.72,REG[key],fill_alpha=.23)
        mask=(peakt>=start)&(peakt<min(start+48,168))
        if ex_vivo:mask&=peakt<0
        points=peakt[mask]-start
        ax.plot(points,np.full(len(points),yy+.77),'o',ms=1.3,color='black')
    ax.plot([0,24,24,48],[6.87,6.87,7.87,7.87],ls='--',color='#666666',lw=.65)
    ax.set(xlim=(0,48),ylim=(-.4,21),xticks=[0,24,48])
    ax.set_yticks([20,7,6,0],['S1','S14','DD1','DD7'] if show_ylabels else ['']*4)
    ax.tick_params(length=0,labelsize=5.8,pad=1)
    for spine in ax.spines.values():spine.set_visible(False)
    if show_xlabel:ax.set_xlabel('Time (h)',labelpad=1,fontsize=6.5)
    if show_condition:ax.text(.5,1.025,condition.replace(' LD',''),transform=ax.transAxes,ha='center',fontsize=7,weight='bold')
    return ax

def dynamic_legend(fig,bbox=(.53,.008),fontsize=5.8):
    handles=[Line2D([],[],color=REG['D'],label='D'),Line2D([],[],color=REG['V'],label='V'),Line2D([],[],color=REG['X'],ls=':',label='X signal'),Line2D([],[],color='black',marker='o',ls='',ms=2.7,label='Behavior peak'),Patch(facecolor='#F6C85F',alpha=.35,label='Scheduled light')]
    return fig.legend(handles=handles,loc='lower center',bbox_to_anchor=bbox,ncol=5,frameon=False,fontsize=fontsize,handlelength=1.4,columnspacing=1.1)
