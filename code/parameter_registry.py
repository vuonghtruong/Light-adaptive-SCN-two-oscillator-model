"""Single source of selected parameters for all current reference analyses."""
from pathlib import Path
import json,hashlib

ROOT=Path(__file__).resolve().parents[1]

def selected(architecture='D-period-X'):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'data/optimization').glob(f'seed_*/fit_{architecture}_main_protocol.json'))]
    if not records:raise FileNotFoundError('Missing selected optimization records: '+architecture)
    if any(q.get('selection_criterion')!='shared_observable_penalty_v2' for q in records):
        raise RuntimeError('Run code/reselect_optimization.py to apply the unified penalized criterion before generating model data.')
    return min(records,key=lambda q:q['final_diagnostic_score'])

def reference_parameters():
    q=selected();p=dict(q['parameters'])
    p['tau_v']=p.pop('tau_v0');p['gphi_d']=p.pop('gphi_s');p['gp_d']=p.pop('gp_s')
    p.update(phase_bias=0.,mem_tau_d=240.,mem_tau_x=240.)
    return p

def fingerprint(architecture='D-period-X'):
    q=selected(architecture)
    return hashlib.sha256(json.dumps({'architecture':architecture,'seed':q['seed'],'vector':q['vector'],'criterion':q['selection_criterion']},sort_keys=True).encode()).hexdigest()
