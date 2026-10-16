from common import *
import ast,hashlib,platform,importlib.metadata,re,xml.etree.ElementTree as ET
import fit_protocol as fit
import parameter_registry

ARCHITECTURES=['D-period-X','coupling-X','V-period-X','joint-DV-period-X']

def frames_equal(actual,expected):
 """CSV round-trips preserve all fields, allowing only numerical serialization."""
 try:
  pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
  return True
 except AssertionError:return False


def historical_coupling_checks():
 """Validate the fixed-KVD historical construction used in current S6."""
 import simple_coupling_rule as rule
 report=json.loads((ROOT/'validation/simple_coupling_rule.json').read_text());source=json.loads((DATA/'simple_coupling_input.json').read_text(encoding='utf-8-sig'))
 cached=json.loads((DATA/'simple_coupling_rule_results.json').read_text());tables={name:pd.read_csv(TABLES/name) for name in rule.TABLE_NAMES};checks=[]
 def add(test,passed,**details):checks.append(dict(test=test,passed=bool(passed),**details))
 hashes=report.get('output_hashes',{})
 add('S6 historical construction fingerprint, hashes and immutable source record',report.get('fingerprint')==rule.fingerprint() and bool(hashes) and all((DATA/name).exists() and hashlib.sha256((DATA/name).read_bytes()).hexdigest()==value for name,value in hashes.items()) and report.get('source_record_sha256')==source['source_sha256'] and report.get('source_seed')==source['historical_selected']['seed'] and report.get('refitted') is False)
 add('S6 three public rule tables exactly reproduce cached records',set(cached['tables'])==set(rule.TABLE_NAMES) and all(frames_equal(table,pd.DataFrame(cached['tables'][name])) for name,table in tables.items()) and [len(tables[name]) for name in rule.TABLE_NAMES]==[5,10,12])
 p=tables['simple_coupling_rule_parameters.csv'].set_index('condition').loc[ORDER];v=np.asarray(source['historical_selected']['vector']);shared=v[[0,1,3,4,5]];pairs=np.column_stack([v[6:],np.repeat(v[2],5)])
 expected=np.column_stack([pairs[:,0],pairs[:,0]+shared[2],pairs[:,1],np.repeat(shared[2],5),pairs.sum(axis=1),pairs.sum(axis=1)+shared[2],np.repeat(2*np.pi/shared[0]-2*np.pi/shared[1],5)])
 actual=p[['K_DV_exvivo_rad_h','K_DV_intact_rad_h','K_VD_rad_h','epsilon_rad_h','K_eff_exvivo_rad_h','K_eff_intact_rad_h','delta_omega_rad_h']].to_numpy()
 add('S6 assigned coefficients reproduce the historical vector with one fixed KVD',np.allclose(actual,expected,rtol=1e-12,atol=1e-12) and p.K_VD_rad_h.nunique()==1 and np.array_equal(report['shared_parameters'],shared) and np.array_equal(report['coupling_pairs'],pairs) and p.source_seed.eq(report['source_seed']).all())
 diagnostics=tables['simple_coupling_rule_diagnostics.csv'];error=0.;valid=True
 for family,predictor,conditions in [('photoperiod','light_h',['SP','LD','LP']),('T-cycle','cycle_h',['T22','LD','T26'])]:
  x=p.loc[conditions,predictor].to_numpy()
  for coefficient,column in [('K_DV','K_DV_exvivo_rad_h'),('K_VD','K_VD_rad_h')]:
   y=p.loc[conditions,column].to_numpy();line=y[0]+(y[2]-y[0])*(x-x[0])/(x[2]-x[0]);z=diagnostics[diagnostics.family.eq(family)&diagnostics.coefficient.eq(coefficient)].set_index('condition').loc[conditions]
   error=max(error,float(abs(z[['predictor_h','actual_rad_h','endpoint_linear_rad_h','residual_rad_h']].to_numpy()-np.column_stack([x,y,line,y-line])).max()))
   slopes=[(y[1]-y[0])/(x[1]-x[0]),(y[2]-y[1])/(x[2]-x[1]),(y[2]-y[0])/(x[2]-x[0])]
   span=abs(y[2]-y[0]);relative=abs(y[1]-line[1])/span if span>1e-14 else 0.
   valid &= z.predictor.eq(predictor).all() and np.allclose(z[['left_slope_rad_h2','right_slope_rad_h2','endpoint_slope_rad_h2']].to_numpy(),np.tile(slopes,(3,1)),rtol=1e-12,atol=1e-12) and np.allclose(z.midpoint_relative_to_endpoint_span,relative,rtol=1e-12,atol=1e-12)
 add('S6 straight references use outer conditions and retain actual LD residuals',valid and error<1e-12,maximum_diagnostic_difference=error)
 phot=float(p.loc['LD','K_DV_exvivo_rad_h']-(p.loc['SP','K_DV_exvivo_rad_h']+p.loc['LP','K_DV_exvivo_rad_h'])/2);cycle=float(p.loc['LD','K_DV_exvivo_rad_h']-(p.loc['T22','K_DV_exvivo_rad_h']+p.loc['T26','K_DV_exvivo_rad_h'])/2);rr=report['rule_diagnostics']
 add('S6 photoperiod and T-cycle nonlinearity summaries match assigned coefficients',np.allclose([phot,cycle,abs(cycle/phot)],[rr['photoperiod_K_DV_LD_residual_rad_h'],rr['Tcycle_K_DV_LD_residual_rad_h'],rr['absolute_residual_ratio_Tcycle_to_photoperiod']],rtol=1e-12,atol=1e-12),photoperiod_LD_residual_rad_h=phot,Tcycle_LD_residual_rad_h=cycle,absolute_residual_ratio=abs(cycle/phot))
 endpoints=tables['simple_coupling_rule_endpoints.csv'];prediction=np.empty((5,2,3))
 for i,c in enumerate(ORDER):
  for j,context in enumerate(['invivo','exvivo']):prediction[i,j]=endpoints[endpoints.condition.eq(c)&endpoints.context.eq(context)].iloc[0][['D_h','V_h','behavior_h']].to_numpy(dtype=float)
 cc=rule.model.contrasts(prediction)
 add('S6 historical spectral orderings and minimum margin reproduce endpoints',bool((cc>0).all())==report['all_spectral_orderings'] and abs(cc.min()-report['minimum_spectral_contrast_h'])<1e-10)
 tr=read('simple_coupling_rule_traces.csv.gz');timing=True
 for c in ORDER:
  a=tr[tr.condition.eq(c)&tr.context.eq('invivo')];b=tr[tr.condition.eq(c)&tr.context.eq('exvivo')];pre=a.time_from_DD_h<0;epre=b.time_from_DD_h<0
  timing &= np.array_equal(a.loc[pre,['phi_D','phi_V']].to_numpy(),b.loc[epre,['phi_D','phi_V']].to_numpy()) and np.allclose(a.X_input,shared[2],rtol=0,atol=1e-15) and np.allclose(b.loc[epre,'X_input'],shared[2],rtol=0,atol=1e-15) and b.loc[~epre,'X_input'].eq(0).all()
 add('S6 stored histories retain systemic offset throughout LD and remove it only in explant DD',timing)
 convergence=report['step_halving'];maxstep=max(r['maximum_endpoint_half_step_difference_h'] for r in convergence)
 add('S6 historical construction passes all phase-equation and numerical checks',report.get('passed',False) and len(report['checks'])==22 and all(r['passed'] for r in report['checks']) and len(convergence)==10 and all(r['passed'] for r in convergence) and maxstep<.005,maximum_endpoint_difference_h=maxstep)
 return checks


def new_supplement_checks():
 """Verify S5/S6 source records, public tables and scientific summaries."""
 import memory_timescale, simple_coupling
 checks=[]
 def add(name,passed,**details):checks.append(dict(test=name,passed=bool(passed),**details))
 memory=json.loads((ROOT/'validation/memory_timescale.json').read_text())
 simple=json.loads((ROOT/'validation/simple_coupling.json').read_text())
 hashes=lambda records: bool(records) and all((DATA/name).exists() and hashlib.sha256((DATA/name).read_bytes()).hexdigest()==value for name,value in records.items())
 add('S5 current selected D parameters, generator and output hashes',memory.get('fingerprint')==memory_timescale.fingerprint() and memory.get('parameter_fingerprint')==parameter_registry.fingerprint() and memory.get('fixed_parameters')==parameter_registry.reference_parameters() and hashes(memory.get('data_sha256',{})))
 ends=read('memory_timescale_endpoints.csv');scores=read('memory_timescale_scores.csv');step=read('memory_timescale_step_halving.csv')
 coverage=(len(ends)==210 and len(scores)==21 and len(ends.drop_duplicates(['mode','varied_tau_h','condition','ex_vivo']))==210 and set(ends['mode'])==set(memory_timescale.MODES) and set(ends.varied_tau_h)==set(memory_timescale.TIMES) and all(len(g)==10 and set(g.condition)==set(ORDER) and set(g.ex_vivo)=={False,True} for _,g in ends.groupby(['mode','varied_tau_h'])))
 add('S5 complete three-mode seven-timescale five-condition/context coverage',coverage and memory.get('endpoint_rows')==210 and memory.get('score_rows')==21 and memory.get('unique_full_simulations')==190)
 consistent=True;score_error=0.;contrast_error=0.
 for _,row in scores.iterrows():
  q=ends[ends['mode'].eq(row['mode'])&ends.varied_tau_h.eq(row.varied_tau_h)];arrays=[]
  for ex in [False,True]:
   z=q[q.ex_vivo.eq(ex)].set_index('condition').loc[ORDER];arr=np.zeros((5,10));arr[:,:6]=z[['behavior_discrete_h','D_discrete_h','V_discrete_h','X_discrete_h','gap_release_folded_h','gap_7day_folded_arithmetic_h']].to_numpy();arrays.append(arr)
  score_error=max(score_error,abs(fit.final_score(*arrays)-row.common_penalized_objective))
  inv=q[~q.ex_vivo].set_index('condition');exv=q[q.ex_vivo].set_index('condition')
  calculated=[inv.loc['T22','behavior_h']-inv.loc['T26','behavior_h'],exv.loc['T22','D_h']-exv.loc['T26','D_h'],exv.loc['T22','V_h']-exv.loc['T26','V_h']]
  recorded=row[['behavior_T22_minus_T26_h','exvivo_D_T22_minus_T26_h','exvivo_V_T22_minus_T26_h']].to_numpy(dtype=float)
  contrast_error=max(contrast_error,float(abs(recorded-calculated).max()))
  consistent &= row.inversion_both_regions==bool(calculated[0]<0 and calculated[1]>0 and calculated[2]>0)
  for region in ['D','V']:consistent &= row[f'exvivo_{region}_order_T22_LD_T26']==bool(exv.loc['T22',region+'_h']>exv.loc['LD',region+'_h']>exv.loc['T26',region+'_h'])
 add('S5 scores and inversion summaries recompute from their endpoints',consistent and score_error<1e-6 and contrast_error<1e-10,maximum_score_difference=score_error,maximum_contrast_difference_h=contrast_error)
 baseline=scores[scores.varied_tau_h.eq(240)].common_penalized_objective.to_numpy();baseline_error=float(abs(baseline-parameter_registry.selected()['final_diagnostic_score']).max())
 add('S5 240 h reproduces the common selected-model objective',len(baseline)==3 and baseline_error<1e-6 and abs(memory['baseline_objective']-baseline[0])<1e-6,maximum_score_difference=baseline_error)
 add('S5 thirty step-halving checks cover extreme and reference constants',len(step)==30 and set(step.memory_tau_h)=={60.,240.,720.} and step.passed.all() and step.maximum_period_difference_h.max()<.005 and step.circular_gap_difference_h.max()<.05 and memory.get('step_halving_cases')==30 and memory.get('passed',False),maximum_period_difference_h=float(step.maximum_period_difference_h.max()),maximum_circular_gap_difference_h=float(step.circular_gap_difference_h.max()))
 add('S5 public tables reproduce the cached scientific inputs',frames_equal(pd.read_csv(TABLES/'memory_timescale_endpoints.csv'),ends) and frames_equal(pd.read_csv(TABLES/'memory_timescale_summary.csv'),scores))
 add('Retained independent-pair control generator, inputs and output hashes',simple.get('fingerprint')==simple_coupling.fingerprint() and hashes(simple.get('output_hashes',{})))
 cached=json.loads((DATA/'simple_coupling_results.json').read_text());tables={name:pd.read_csv(TABLES/name) for name in simple_coupling.OUTPUTS}
 add('Retained independent-pair control eight public tables reproduce cached records',set(cached['tables'])==set(simple_coupling.OUTPUTS) and all(frames_equal(table,pd.DataFrame(cached['tables'][name])) for name,table in tables.items()))
 params=tables['simple_coupling_parameters.csv'].set_index('condition').loc[ORDER];shared=tables['simple_coupling_shared_parameters.csv'].set_index('parameter');s=np.asarray(cached['shared']);pairs=np.asarray(cached['couplings'])
 expected=np.column_stack([pairs[:,0],pairs[:,0]+s[2],pairs[:,1],np.repeat(s[2],5),pairs.sum(axis=1),pairs.sum(axis=1)+s[2],np.repeat(2*np.pi/s[0]-2*np.pi/s[1],5)])
 actual=params[['K_DV_exvivo_rad_h','K_DV_intact_rad_h','K_VD_rad_h','epsilon_rad_h','K_eff_exvivo_rad_h','K_eff_intact_rad_h','delta_omega_rad_h']].to_numpy()
 add('Retained independent-pair coefficients and shared parameters match simulations',params.shape[0]==5 and len(shared)==9 and np.allclose(actual,expected,rtol=1e-12,atol=1e-12) and np.allclose(shared.loc[['tau_D','tau_V','epsilon','A_D','A_V'],'value'],s,rtol=1e-12,atol=1e-12) and np.array_equal(s,simple['shared_parameters']) and np.array_equal(pairs,simple['coupling_pairs']))
 grid=tables['simple_coupling_grid.csv'];raw=tables['simple_coupling_sensitivity.csv'];summary=tables['simple_coupling_perturbation_summary.csv'];spectral=pd.DataFrame(cached['spectral_validation'])
 add('Retained independent-pair local grid classifications and counts reproduce',len(grid)==41*41 and grid.K_DV_common_offset_rad_h.nunique()==41 and grid.epsilon_rad_h.nunique()==41 and (grid.all_orderings==(grid.minimum_contrast_h>0)).all() and int(grid.all_orderings.sum())==simple['grid_retained'] and len(grid)==simple['grid_total'],retained=int(grid.all_orderings.sum()),total=len(grid))
 calculated=raw.groupby('relative_radius').agg(retained=('all_orderings','sum'),total=('all_orderings','size'),retention_fraction=('all_orderings','mean'),margin_0_01h_fraction=('all_margin_0_01h','mean')).reset_index()
 calculated=calculated.merge(spectral.groupby('relative_radius').agg(spectral_checked=('replicate','size'),spectral_retained=('spectral_orderings','sum'),spectral_retention_fraction=('spectral_orderings','mean')).reset_index())
 add('Retained independent-pair perturbation fractions reproduce all samples',len(raw)==2560 and len(summary)==5 and set(raw.relative_radius)=={.01,.02,.05,.1,.2} and (raw.all_orderings==(raw.minimum_contrast_h>0)).all() and frames_equal(summary,calculated) and frames_equal(summary,pd.DataFrame(simple['perturbation_summary'])))
 convergence=tables['simple_coupling_convergence.csv'];maxstep=float(abs(convergence.iloc[:,2:]).max().max())
 add('Retained independent-pair dynamics and ten step-halving checks passed',simple.get('passed',False) and len(simple['checks'])==21 and all(q['passed'] for q in simple['checks']) and len(convergence)==10 and maxstep<.005,maximum_endpoint_difference_h=maxstep)
 endpoints=tables['simple_coupling_endpoints.csv'];prediction=np.empty((5,2,3))
 for i,c in enumerate(ORDER):
  for j,context in enumerate(['invivo','exvivo']):prediction[i,j]=endpoints[endpoints.condition.eq(c)&endpoints.context.eq(context)].iloc[0][['D_h','V_h','behavior_h']].to_numpy(dtype=float)
 cc=simple_coupling.contrasts(prediction)
 add('Retained independent-pair endpoint orderings and spectral margin recompute',bool((cc>0).all())==simple['all_spectral_orderings'] and abs(float(cc.min())-simple['minimum_spectral_contrast_h'])<1e-10 and len(spectral)==simple['spectral_validation_n'] and int((spectral.regression_orderings==spectral.spectral_orderings).sum())==simple['spectral_validation_agreements'])
 fitted=pd.read_csv(TABLES/'Table_S3_fitted_parameters.csv');params_ok=len(fitted)==65 and set(fitted.architecture)==set(ARCHITECTURES)
 for arch in ARCHITECTURES:
  selected=parameter_registry.selected(arch);g=fitted[fitted.architecture.eq(arch)].set_index('parameter')
  params_ok &= len(g)==len(selected['parameters']) and set(g.index)==set(selected['parameters'])
  for name,value in selected['parameters'].items():
   r=g.loc[name];params_ok &= r.selected_seed==selected['seed'] and r.role=='fitted' and np.isclose(r.value,value,rtol=1e-12,atol=1e-12) and np.allclose([r.lower_bound,r.upper_bound],selected['bounds'][name],rtol=1e-12,atol=1e-12) and np.isclose(r.shared_final_score,selected['final_diagnostic_score'],rtol=1e-12,atol=1e-12)
 add('Table S3 all 65 fitted values, bounds, seeds and scores match selected registry',params_ok)
 fixed=pd.read_csv(TABLES/'Table_S3_fixed_model_settings.csv').set_index('parameter');reference=parameter_registry.reference_parameters()
 constants={'SCN memory e-folding time':reference['mem_tau_d'],'Systemic memory e-folding time':reference['mem_tau_x'],'Behavioral D weight':.8,'SCN-to-X D weight':.8,'Initial D phase':.35,'Initial V phase':0.,'Initial X phase':-.2,'Initial memories':0.,'Light exposure duration':720.,'DD duration':336.,'Primary/final integration step':fit.DT_FINAL,'Saved signal sampling interval':.25,'Screening/initial-ensemble step':fit.DT,'Adaptive-period lower clip':16.,'Adaptive-period upper clip':34.}
 add('Table S3 all 15 fixed settings match the simulation and fitting protocol',len(fixed)==15 and set(fixed.index)==set(constants) and all(np.isclose(fixed.loc[name,'value'],value,rtol=0,atol=1e-12) for name,value in constants.items()))
 checks.extend(historical_coupling_checks())
 return checks,memory,simple


def main_summary_point_checks():
 """Every displayed ensemble point represents one stored initial-phase run."""
 report=json.loads((ROOT/'validation/main_summary_display.json').read_text());checks=[]
 for figure,filename,conditions,panel in [('Figure_3','perturbation_initial_ensemble.csv',ORDER,'Figure 3F'),('Figure_4','original_logic_prediction_ensemble.csv',['18:6 LD','T23','T25'],'Figure 4D')]:
  source=read(filename);points=report[figure]['period_ensemble_points'];valid=len(points)==3*len(conditions);error=0.
  expected={(c,False,'behavior_h') for c in conditions}|{(c,True,k) for c in conditions for k in ['D_h','V_h']}
  valid &= {(r['condition'],r['ex_vivo'],r['source_column']) for r in points}==expected
  for row in points:
   values=source[source.condition.eq(row['condition'])&source.ex_vivo.eq(row['ex_vivo'])][row['source_column']].dropna()
   error=max(error,abs(values.min()-row['minimum_h']),abs(values.max()-row['maximum_h']))
   valid &= row['available_rows']==len(values)==row['drawn_points']==64 and row['horizontal_jitter_only'] is True and row['point_color']=='black'
  checks.append(dict(test=figure+' period point series use all correct source rows and horizontal-only jitter',passed=bool(valid and error<1e-10),maximum_source_range_difference_h=float(error)))
  behavioral=[r for r in points if not r['ex_vivo'] and r['source_column']=='behavior_h'];count=sum(r['drawn_points'] for r in behavioral)
  checks.append(dict(test=panel+' includes all 64 behavioral initial states per schedule',passed=count==64*len(conditions) and len(behavioral)==len(conditions),behavioral_point_count=count))
 return checks



def observable_components(inv,ex):
 """Readable additive decomposition of the complete shared observable score."""
 names=['behavior_target_fit','behavior_tolerance','photoperiod_context_split','Tcycle_exvivo_absolute_targets','Tcycle_context_inversion','phase_target_fit','phase_tolerance','behavior_order_and_contrasts','exvivo_photoperiod_order','phase_direction_and_size','T22_regional_order','Tcycle_exvivo_order']
 parts=dict.fromkeys(names,0.)
 def add(key,value):parts[key]+=float(value)
 for i in range(5):
  err=abs(inv[i,0]-fit.BT[i]);tol=.14 if i<3 else .25
  add('behavior_target_fit',(600 if i<3 else 500)*(err/(.08 if i<3 else .18))**2)
  if err>tol:add('behavior_tolerance',3000000+1200000*(err-tol)**2)
  for j in (1,2):
   if i<3:add('photoperiod_context_split',1400*((ex[i,j]-inv[i,j])/.08)**2)
   else:
    add('Tcycle_exvivo_absolute_targets',700*((ex[i,j]-fit.ET[i])/.16)**2)
    split=ex[i,j]-inv[i,j] if i==3 else inv[i,j]-ex[i,j]
    if split<.70:add('Tcycle_context_inversion',2500000+1200000*(.70-split)**2)
  if i<3:
   split=max(abs(ex[i,1]-inv[i,1]),abs(ex[i,2]-inv[i,2]))
   if split>.22:add('photoperiod_context_split',9000000+4000000*(split-.22)**2)
  for j in (4,5):add('phase_target_fit',140*((ex[i,j]-fit.GT[i])/fit.GS[i])**2)
  err=max(abs(ex[i,4]-fit.GT[i]),abs(ex[i,5]-fit.GT[i]));tol=.45 if i<3 else .70
  if err>tol:add('phase_tolerance',200000+150000*(err-tol)**2)
 if not inv[1,0]<inv[2,0]<inv[0,0]:add('behavior_order_and_contrasts',1000000)
 if inv[2,0]-inv[1,0]<.09:add('behavior_order_and_contrasts',1000000+600000*(.09-inv[2,0]+inv[1,0])**2)
 if inv[0,0]-inv[2,0]<.18:add('behavior_order_and_contrasts',1000000+600000*(.18-inv[0,0]+inv[2,0])**2)
 if not inv[3,0]<inv[2,0]<inv[4,0]:add('behavior_order_and_contrasts',1000000)
 add('behavior_order_and_contrasts',5000*((inv[0,0]-inv[2,0])-.27)**2+5000*((inv[2,0]-inv[1,0])-.11)**2)
 for j in (0,1,2):
  if not ex[1,j]<ex[2,j]<ex[0,j]:add('exvivo_photoperiod_order',1000000)
  if ex[2,j]-ex[1,j]<.08:add('exvivo_photoperiod_order',3500+3000*(.08-ex[2,j]+ex[1,j])**2)
  if ex[0,j]-ex[2,j]<.12:add('exvivo_photoperiod_order',3500+3000*(.12-ex[0,j]+ex[2,j])**2)
 if ex[3,5]>=-1.20:add('phase_direction_and_size',6000000+2000000*(ex[3,5]+1.20)**2)
 for i in (0,1,2,4):
  if ex[i,5]<=0:add('phase_direction_and_size',1000000+100000*ex[i,5]**2)
 if ex[4,5]<1.40:add('phase_direction_and_size',3500000+1600000*(1.40-ex[4,5])**2)
 if ex[3,5]>=0:add('phase_direction_and_size',8000000+2500000*ex[3,5]**2)
 diff=ex[3,1]-ex[3,2]
 if diff<=0:add('T22_regional_order',8000000+2500000*diff**2)
 elif diff<.18:add('T22_regional_order',1500000+800000*(.18-diff)**2)
 for j in (1,2):
  if ex[3,j]<=ex[2,j]+.20:add('Tcycle_exvivo_order',5000+1000*(ex[2,j]+.20-ex[3,j])**2)
  if ex[4,j]>=ex[2,j]-.20:add('Tcycle_exvivo_order',1000000+100000*(ex[4,j]-ex[2,j]+.20)**2)
 return parts

def target_error_tables():
 cases=[('original_reference','D-period-X','original',fit.archive_vector())]
 cases += [(a,a,parameter_registry.selected(a)['seed'],np.asarray(parameter_registry.selected(a)['vector'])) for a in ARCHITECTURES]
 rows=[];parts=[];differences=[]
 for label,arch,seed,vector in cases:
  inv,ex,_=fit.rescore(vector,arch)
  components=observable_components(inv,ex);score=float(fit.final_score(inv,ex));differences.append(abs(sum(components.values())-score))
  for name,value in components.items():parts.append(dict(parameter_set=label,architecture=arch,seed=seed,component=name,value=value,total_score=score))
  for i,c in enumerate(ORDER):
   definitions=[('behavior',inv[i,0],fit.BT[i],float(fit.TARGETS.iloc[i].behavior_sem_h),'direct behavioral target'),('exvivo_D',ex[i,1],fit.ET[i],np.nan,'direct T-cycle target penalty' if i>=3 else 'displayed auxiliary target; objective constrains ex/in vivo split'),('exvivo_V',ex[i,2],fit.ET[i],np.nan,'direct T-cycle target penalty' if i>=3 else 'displayed auxiliary target; objective constrains ex/in vivo split'),('gap_release_folded',ex[i,4],fit.GT[i],fit.GS[i],'direct folded phase target'),('gap_first7day_folded',ex[i,5],fit.GT[i],fit.GS[i],'direct folded phase target')]
   for observable,prediction,target,sem,role in definitions:rows.append(dict(parameter_set=label,architecture=arch,seed=seed,condition=c,observable=observable,predicted_h=prediction,target_h=target,residual_h=prediction-target,absolute_error_h=abs(prediction-target),recorded_target_sem_h=sem,objective_role=role))
 pd.DataFrame(rows).to_csv(TABLES/'fit_target_errors.csv',index=False)
 pd.DataFrame(parts).to_csv(TABLES/'fit_objective_components.csv',index=False)
 return max(differences)


def main():
 checks=[];runs=[]
 for arch in ARCHITECTURES:
  qs=[json.loads(p.read_text()) for p in sorted((DATA/'optimization').glob(f'seed_*/fit_{arch}_main_protocol.json'))]
  checks.append({'test':arch+' has three paired seeds','passed':sorted(q['seed'] for q in qs)==[1729,2718,3141]})
  for q in qs:
   checks.append({'test':f'{arch} seed {q["seed"]} unified selection criterion','passed':q.get('selection_criterion')=='shared_observable_penalty_v2'})
   z=pd.DataFrame(q['selected']['endpoints']);arrays=[]
   for ex in [False,True]:
    g=z[z.ex_vivo.eq(ex)].set_index('condition').loc[ORDER];arr=np.zeros((5,10));arr[:,:6]=g[['behavior_h','D_h','V_h','X_h','gap_release_folded_h','gap_7day_folded_arithmetic_h']].to_numpy();arrays.append(arr)
   score=fit.final_score(*arrays)
   checks.append({'test':f'{arch} seed {q["seed"]} score recomputation','passed':abs(score-q['final_diagnostic_score'])<1e-9,'absolute_difference':abs(score-q['final_diagnostic_score'])})
   runs.append({k:q[k] for k in ['architecture','seed','free_parameter_count','latin_samples','local_starts','maximum_local_iterations','final_candidates','evaluations_screening','elapsed_seconds','final_diagnostic_score','selected_origin']})
 for seed in [1729,2718,3141]:
  q=[json.loads((DATA/'optimization'/f'seed_{seed}'/f'fit_{a}_main_protocol.json').read_text()) for a in ['D-period-X','coupling-X']]
  shared=set(q[0]['bounds'])&set(q[1]['bounds']);checks.append({'test':f'seed {seed} equal shared bounds','passed':all(q[0]['bounds'][k]==q[1]['bounds'][k] for k in shared),'shared_parameters':len(shared)})
  checks.append({'test':f'seed {seed} equal search ceilings','passed':all(q[0][k]==q[1][k] for k in ['latin_samples','local_starts','maximum_local_iterations','free_parameter_count'])})
 pd.DataFrame(runs).to_csv(TABLES/'Table_S2_matched_searches.csv',index=False)
 # Re-run the selected forward models after making the code portable.
 for arch in ARCHITECTURES:
  q=chosen(arch);iv,ev,endpoints=fit.rescore(np.array(q['vector']),arch)
  cols=['behavior_h','D_h','V_h','gap_release_folded_h','gap_7day_folded_arithmetic_h']
  old=pd.DataFrame(q['selected']['endpoints']).set_index(['condition','ex_vivo'])[cols]
  new=pd.DataFrame(endpoints).set_index(['condition','ex_vivo'])[cols].reindex(old.index)
  error=float(abs(new-old).to_numpy().max())
  checks.append({'test':arch+' selected forward-model reproduction','passed':error<1e-8,'maximum_endpoint_difference':error})
 # Independent algebraic-lag integrations, not replayed original trajectories.
 import reduced_model as reduced
 reduction_parameters=json.loads((DATA/'reduction_parameters.json').read_text())
 expected=read('reduction_targets.csv')
 checks.append({'test':'Reduction uses exactly current D and coupling winners','passed':len(reduction_parameters)==2 and all(p['vector']==parameter_registry.selected(p['architecture'])['vector'] for p in reduction_parameters)})
 for p,(condition,ex) in zip(reduction_parameters,[('T22',False),('T26',True)]):
  tr,invalid_ld,invalid_dd=reduced.run(np.array(p['vector']),reduced.original.ARCHITECTURES[p['architecture']],*reduced.original.SCHEDULES[condition],.015625,ex,True)
  values,_=reduced.metrics(tr)
  r=expected[expected.fit_id.eq(p['fit_id'])&expected.variant.eq('adiabatic_coherent')&expected.condition.eq(condition)&expected.context.eq('exvivo' if ex else 'invivo')].iloc[0]
  error=float(abs(values[:6]-r[['behavior_h','D_h','V_h','X_h','gap_release_h','gap_7day_h']].to_numpy(dtype=float)).max())
  checks.append({'test':p['fit_id']+' independent reduction reproduction','passed':error<1e-8 and invalid_ld+invalid_dd==0,'maximum_endpoint_difference':error})
 # Empirical provenance is checked independently of the plotted k labels.
 t=pd.read_csv(TABLES/'Table_S1_literature_provenance.csv');checks.append({'test':'Table S1 keeps all 102 archived observations','passed':len(t)==102});checks.append({'test':'Figure 1 selects 37 SCN means','passed':int(((t.measurement=='SCN')&t.selected_for_Figure1).sum())==37})
 pairs=pd.read_csv(TABLES/'Figure1_plotted_pairs.csv')
 known=pairs.reporter_behavior.notna() & ~pairs.reporter_behavior.fillna('').str.lower().isin(['not specified','not reported','unknown',''])
 corrected=pairs[pairs.study.eq('Myung et al.')&pairs.year.eq(2012)&pairs.reporter_behavior.eq('Bmal1-ELuc')]
 checks.append({'test':'Figure 1 reporter-safe pairing and Myung2012 correction','passed':bool(len(pairs)==31 and pairs.loc[known,'reporter_behavior'].eq(pairs.loc[known,'reporter_scn']).all() and len(corrected)==1 and np.isclose(corrected.iloc[0].scn_mean_period_h,23.9) and np.isclose(corrected.iloc[0].behavior_mean_period_h,23.7))})
 groups=pd.read_csv(TABLES/'Figure1_regression_groups.csv');audit=json.loads((ROOT/'validation/figure1_audit.json').read_text())
 checks.append({'test':'Figure 1 complete eligible fits and explicitly descriptive two-mean connectors','passed':bool(len(groups)==14 and int(groups.n.ge(3).sum())==5 and int(groups.n.eq(2).sum())==7 and groups.loc[groups.n.lt(3),'pearson_r'].isna().all() and audit['regression_count']==5 and audit['two_mean_connector_count']==7)})
 checks.append({'test':'Figure 1 categorical error bars remain visible','passed':all(r['clipped_errorbars']==0 for r in audit['categorical_errorbar_clipping'])})
 tr=read('full_model_dynamic_traces.csv.gz')
 checks.append({'test':'All main dynamic traces are full original model','passed':tr.model.eq('full_original_D_V_X').all().item()})
 for c in ORDER+['18:6 LD','T23','T25']:
  a=tr[tr.condition.eq(c)&tr.context.eq('invivo')].set_index('time_from_DD_h');b=tr[tr.condition.eq(c)&tr.context.eq('exvivo')].set_index('time_from_DD_h');cols=['D','V','X','phi_D','phi_V','phi_X','memory_D','memory_X','X_to_D_rad_h','X_to_V_rad_h']
  pre=float(np.max(abs(a.loc[a.index<0,cols].to_numpy()-b.loc[b.index<0,cols].to_numpy())));post=float(np.max(abs(b.loc[b.index>=0,['X_to_D_rad_h','X_to_V_rad_h']].to_numpy())))
  checks.append({'test':c+' pre-DD intact/explant equality and DD feedback removal','passed':pre==0 and post==0,'pre_error':pre,'DD_feedback':post})
 import architecture_sensitivity as sensitivity
 sweep=read('Figure2_full_model_gain_sweep.csv')
 meta=json.loads((ROOT/'validation/Figure2_gain_sweep_metadata.json').read_text())
 coverage=(len(sweep)>=810 and sweep.groupby(['architecture','condition']).gain_scale.nunique().ge(81).all()
           and np.isclose(sweep.gain_scale.min(),0) and np.isclose(sweep.gain_scale.max(),2))
 coverage &= all(all(np.isclose(g.gain_scale,s).any() for s in np.linspace(0,2,81)) for _,g in sweep.groupby(['architecture','condition']))
 checks.append({'test':'Figure 2 complete full-model gain sweep and current provenance','passed':bool(coverage and meta['fingerprint']==sensitivity.fingerprint())})
 checks.append({'test':'Figure 2 phase wrapping and feedback timing','passed':bool(sweep.gap_circular_7d_h.between(-12,12).all() and sweep.maximum_ex_DD_feedback_rad_h.eq(0).all() and sweep.maximum_preDD_feedback_rad_h.gt(0).all())})
 ratio=abs(sweep.release_delta_omega_rad_h/sweep.release_K_eff_rad_h)
 ratio.loc[abs(sweep.release_K_eff_rad_h)<=1e-10]=np.inf
 checks.append({'test':'Figure 2 release locking diagnostic','passed':bool(np.allclose(ratio,sweep.release_locking_ratio,rtol=1e-10,atol=1e-12,equal_nan=True))})
 cv=pd.read_csv(TABLES/'Figure2_gain_sweep_convergence.csv')
 checks.append({'test':'Figure 2 gain-sweep step halving and selected-fit equality','passed':bool(len(cv)>=30 and cv[['D_period_difference_h','V_period_difference_h']].max().max()<.01 and cv.circular_phase_difference_h.max()<.01 and meta['maximum_baseline_fit_grid_difference_h']<1e-10),'convergence_cases':len(cv)})
 display=json.loads((ROOT/'validation/Figure2_display_checks.json').read_text())
 checks.append({'test':'Figure 2 paired y-axis limits match','passed':all(np.array_equal(v[0],v[1]) for v in display['paired_y_limits'].values())})
 selected_ok=True
 for arch,info in display['selected_fits'].items():
  observed=[r['final_diagnostic_score'] for r in runs if r['architecture']==arch]
  selected_ok &= info['selected_seed']==chosen(arch)['seed'] and np.isclose(info['selected_final_objective'],min(observed),atol=1e-10,rtol=0)
 checks.append({'test':'Figure 2 selected fit is best recorded common objective among three seeds','passed':bool(selected_ok)})
 # Check actual plotted last-strip artists; the DD8 half must contain no line or peak.
 from actogram_style import doubleplot_axis
 display_ok=True
 for condition in ORDER+['18:6 LD','T23','T25']:
  for ex in [False,True]:
   f,ax=plt.subplots();doubleplot_axis(ax,tr,condition,ex)
   for line in ax.lines[-5:-1]:
    x=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata());visible=np.isfinite(y)
    display_ok &= bool((x[visible]<=24+1e-9).all())
   display_ok &= [tick.get_text() for tick in ax.get_yticklabels()]==['S1','S14','DD1','DD7']
   forbidden=np.array([232,239,245])/255
   display_ok &= not any(np.allclose(color[:3],forbidden,atol=1e-5) for collection in ax.collections for color in collection.get_facecolors())
   plt.close(f)
 checks.append({'test':'Shared double plots stop at DD7 and have no DD background','passed':bool(display_ok)})
 contrast=pd.read_csv(TABLES/'Figure3_explant_sensitivity.csv');contrast_error=0.
 for source,kind in [('explant_coupling_sweep.csv','coupling'),('explant_controls.csv','memory_phase')]:
  src=read(source)
  for _,r in contrast[contrast.test.eq(kind)].iterrows():
   mask=np.isclose(src.coupling_scale,r.coupling_percent/100)
   if kind=='memory_phase':mask &= np.isclose(src.memory_D_retained,r.memory_retained_percent/100)&np.isclose(src.D_phase_advance_h,r.D_phase_advance_h)
   a=src[mask&src.condition.eq('T22')].iloc[0];b=src[mask&src.condition.eq('T26')].iloc[0];ld=src[mask&src.condition.eq('LD')].iloc[0]
   contrast_error=max(contrast_error,abs(r.T22_minus_T26_h-(a[r.region+'_h']-b[r.region+'_h'])),abs(r.LD_period_h-ld[r.region+'_h']))
 checks.append({'test':'Figure 3 direct ordering uses identical T22/LD/T26 perturbations','passed':bool(len(contrast)==64 and contrast_error<1e-10),'maximum_difference_h':float(contrast_error)})
 phase_controls=contrast[contrast.test.eq('memory_phase')]
 strict_counts=phase_controls.groupby('region').T22_gt_LD_gt_T26.sum().to_dict()
 checks.append({'test':'T22 remains above T26 across all 27 explant combinations in both regions','passed':bool(len(phase_controls)==54 and phase_controls.T22_gt_T26.all()),'strict_T22_LD_T26_counts':{k:int(v) for k,v in strict_counts.items()}})
 summary=pd.read_csv(TABLES/'Figure3K_absolute_period_summary.csv');raw=read('explant_controls.csv');summary_error=0.;summary_counts=True
 for _,r in summary.iterrows():
  values=raw[raw.condition.eq(r.condition)&np.isclose(raw.memory_D_retained,r.memory_retained_percent/100)][r.region+'_h']
  expected=np.array([values.mean(),values.min(),values.max()]);actual=r[['mean_period_h','min_period_h','max_period_h']].to_numpy(dtype=float)
  summary_error=max(summary_error,float(abs(expected-actual).max()));summary_counts &= len(values)==9 and r.perturbation_combinations==9
 checks.append({'test':'Figure 3K absolute periods and sensitivity ranges reproduce nine matched settings','passed':bool(len(summary)==18 and summary_counts and summary_error<1e-10),'maximum_difference_h':summary_error})
 step=read('explant_step_halving.csv')
 checks.append({'test':'All 81 current-fit explant controls passed period step halving','passed':bool(len(step)==243 and step.absolute_difference_h.max()<.002),'maximum_period_difference_h':float(step.absolute_difference_h.max())})
 full=json.loads((ROOT/'validation/full_model_dynamic_data.json').read_text())
 checks.append({'test':'Full-model novel-condition step halving','passed':full['passed'] and all(r['passed'] for r in full['step_halving'])})
 checks.append({'test':'Stored full-model trajectories match current selected parameter provenance','passed':full.get('parameter_fingerprint')==parameter_registry.fingerprint() and full.get('parameters')==parameter_registry.reference_parameters() and full.get('trace_sha256')==hashlib.sha256((DATA/'full_model_dynamic_traces.csv.gz').read_bytes()).hexdigest()})
 tree=ast.parse((ROOT/'code/reference_model.py').read_text(encoding='utf-8'))
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='integrate')
 ast_sha=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
 checks.append({'test':'Full-original integrate function unchanged after parameter registry update','passed':ast_sha=='34b5bdc7c3fa02019e4b42821edac4033154d6e63ee93a7ad27bd8ff90608e33','integrate_ast_sha256':ast_sha})
 import reference_model
 checks.append({'test':'Full-model simulator uses current selected D parameters','passed':reference_model.PARAMS==parameter_registry.reference_parameters()})
 cp=read('explant_coupling_sweep.csv');retained=True
 for scale in cp.coupling_scale.unique():
  z=cp[cp.coupling_scale.eq(scale)].set_index('condition')
  retained &= all(z.loc['T22',k]>z.loc['LD',k]>z.loc['T26',k] for k in ['D_h','V_h'])
 checks.append({'test':'T22/LD/T26 ordering across five explant coupling scales','passed':bool(retained)})
 reduction=json.loads((ROOT/'validation/reduction_validation.json').read_text())
 import unified_reduction
 checks.append({'test':'Reduction data provenance matches current selected fits and generator','passed':reduction.get('fingerprint')==unified_reduction.fingerprint() and all(hashlib.sha256((DATA/name).read_bytes()).hexdigest()==value for name,value in reduction['output_sha256'].items())})
 checks.append({'test':'Current-fit independent-reduction numerical validation (not equivalence of reduced and full targets)','passed':reduction['all_checks_passed'] and reduction['simulation_cases']==60 and reduction['half_step_cases']==20})
 extra_checks,memory,simple=new_supplement_checks();checks.extend(extra_checks);checks.extend(main_summary_point_checks())
 from architecture_figures import validate_memory_decomposition
 checks.append(validate_memory_decomposition())
 component_error=target_error_tables()
 checks.append({'test':'Named objective components sum to complete shared final score','passed':component_error<1e-6,'maximum_score_difference':component_error})
 checks.append({'test':'Exactly four main and six numbered supplementary figures','passed':len(list((ROOT/'figures/main').glob('*.svg')))==4 and sorted(p.stem for p in (ROOT/'figures/supplement').glob('*.svg'))==[f'Figure_S{i}' for i in range(1,7)]})
 for p in (ROOT/'figures').rglob('*.svg'):
  r=ET.parse(p).getroot();w=float(r.attrib['width'].replace('pt',''))*25.4/72;t=' '.join(''.join(e.itertext()) for e in r.iter() if e.tag.endswith('}text'))
  checks.append({'test':p.name+' width and no separate H model','passed':abs(w-175)<.01 and not any(s.lower() in t.lower() for s in ['slow H','slowH','D-period-H','H state'])})
  # For math text, use the largest span as the base font, excluding reduced superscripts/subscripts.
  sizes=[max(float(m) for child in e.iter() for m in re.findall(r'font-size:\s*([\d.]+)px',child.attrib.get('style',''))) for e in r.iter() if e.tag.endswith('}text') and any('font-size:' in child.attrib.get('style','') for child in e.iter())]
  checks.append({'test':p.name+' enlarged text','passed':bool(sizes and min(sizes)>=7.2-1e-6),'minimum_base_font_pt':min(sizes) if sizes else None})
  if p.stem=='Figure_3':checks.append({'test':'Figure 3K no unexplained intact-behavior reference','passed':'intact' not in t.lower()})
 report={'passed':all(c['passed'] for c in checks),'checks':checks};(ROOT/'validation/package_checks.json').write_text(json.dumps(report,indent=2))
 # Export numerical control contrasts from the regenerated simulations.
 d=chosen('D-period-X');c=chosen('coupling-X');ctrl=read('fixed_period_release.csv');stat=[]
 for window in ctrl.window.unique():
  z=ctrl[ctrl.window.eq(window)].pivot(index='coupling_scale_before',columns='T_h',values='network_h');v=z[22]-z[26];stat.append(dict(window=window,T22_minus_T26_min_h=v.min(),T22_minus_T26_max_h=v.max()))
 pd.DataFrame(stat).to_csv(TABLES/'fixed_period_Tcycle_contrasts.csv',index=False)
 env={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','matplotlib','astropy','numba']}}
 (ROOT/'validation/environment.json').write_text(json.dumps(env,indent=2))
 print(json.dumps({'passed':report['passed'],'checks':len(checks),'D_score':d['final_diagnostic_score'],'coupling_score':c['final_diagnostic_score']},indent=2))
 if not report['passed']:raise AssertionError([c for c in checks if not c['passed']])
if __name__=='__main__':main()
