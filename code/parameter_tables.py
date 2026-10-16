"""Export selected parameters and fixed settings as numerical CSV tables.

All numeric text comes from the selected parameter registry and generated CSV
tables. No optimization or new literature extraction is performed here.
"""
from pathlib import Path
import json,math,re
import numpy as np
import pandas as pd
import architecture_model as model
import parameter_registry

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'tables'
ARCHES=['D-period-X','coupling-X','V-period-X','joint-DV-period-X']
LABELS={'D-period-X':'Adaptive D','coupling-X':'Adaptive coupling','V-period-X':'Adaptive V','joint-DV-period-X':'Joint D+V'}
SYMBOLS={'tau_d0':'τD,0','tau_v0':'τV,0','tau_x0':'τX,0','g_d':'gD','g_x':'gX','g_v':'gV','g_coupling':'gK','gphi_s':'gφ,SCN','gp_s':'gP,SCN','gphi_x':'gφ,X','gp_x':'gP,X','kvd':'KVD,0 (D→V)','kdv':'KDV (repulsive V→D)','kx_d':'KXD','kx_v':'KXV','ksx':'KSX','light_amp':'Light amplitude','v_light_boost':'V light multiplier'}

def unit(name):
    if name.startswith('tau_') or name in ['g_d','g_x','g_v']:return 'h'
    if name.startswith('k') or name in ['g_coupling','light_amp']:return 'rad/h'
    return 'dimensionless'

def parameter_tables():
    rows=[]
    for arch in ARCHES:
        record=parameter_registry.selected(arch)
        for name in model.NAMES:
            if name not in model.FREE[arch]:continue
            lo,hi=record['bounds'][name]
            rows.append(dict(architecture=arch,selected_seed=record['seed'],parameter=name,
                             symbol=SYMBOLS[name],value=record['parameters'][name],unit=unit(name),
                             lower_bound=lo,upper_bound=hi,role='fitted',
                             shared_final_score=record['final_diagnostic_score']))
    fitted=pd.DataFrame(rows)
    fitted.to_csv(TABLES/'Table_S3_fitted_parameters.csv',index=False)
    constants=[('SCN memory e-folding time',240.,'h','D/V memory where present'),
               ('Systemic memory e-folding time',240.,'h','X memory'),
               ('Behavioral D weight',.8,'dimensionless','Circular phase readout'),
               ('SCN-to-X D weight',.8,'dimensionless','Circular phase feedback'),
               ('Initial D phase',.35,'rad','All primary simulations'),
               ('Initial V phase',0.,'rad','All primary simulations'),
               ('Initial X phase',-.2,'rad','All primary simulations'),
               ('Initial memories',0.,'dimensionless','All active memories'),
               ('Light exposure duration',720.,'h','Before DD release'),
               ('DD duration',336.,'h','Period-estimation record'),
               ('Primary/final integration step',1/64,'h','Forward Euler'),
               ('Saved signal sampling interval',.25,'h','Distinct from integration step'),
               ('Screening/initial-ensemble step',1/16,'h','Coarse screening and phase ensembles'),
               ('Adaptive-period lower clip',16.,'h','D/V/X where adaptive'),
               ('Adaptive-period upper clip',34.,'h','D/V/X where adaptive')]
    fixed=pd.DataFrame(constants,columns=['parameter','value','unit','scope'])
    fixed.to_csv(TABLES/'Table_S3_fixed_model_settings.csv',index=False)
    return fitted,fixed
