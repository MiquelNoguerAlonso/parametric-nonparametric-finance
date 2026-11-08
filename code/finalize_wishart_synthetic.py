"""Promote a complete fixed-parameter fine-grid synthetic evaluation."""
import json,hashlib,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wishart_completion';P=OUT/'primary'
N=131072;S=32
# Validate all ten before writing any primary result.
records=[]
for unit in range(10):
 f=OUT/'results'/f'synthetic_d{unit}_b0_n{N}_s{S}.json';x=json.loads(f.read_text());oldf=ROOT/'wishart_extension/results'/f'heston_{unit}_primary.json';old=json.loads(oldf.read_text());fit=ROOT/'wishart_extension/fits'/f'heston_{unit}.json'
 assert x['paths']==N and x['finest']==S and x['endpoint_max_error']<1e-10
 assert x['parameters']==old['parameters'] and x['parameter_sha256']==hashlib.sha256(fit.read_bytes()).hexdigest()
 ev=copy.deepcopy(old);ev['n_scenarios']=None;ev['n_conditional_paths']=N;ev['substeps']=S;ev['risk_evaluation']='conditional Gaussian mixture tail integration';ev['risk_source']=str(f.relative_to(ROOT));ev['original_record']=str(oldf.relative_to(ROOT));ev['dependence_source']='original one-million-window four-step evaluation';ev['validation_source']='original fixed-grid fit; validation was not used for parameter selection';ev['numerical_refinement_only']=True
 for r in ev['metrics']:
  fine=next(v for v in x['risk'] if v['substeps']==S and v['alpha']==r['alpha']);r['var']=fine['var'];r['es']=fine['es'];r['scenario_mc_se']=fine['mc_se'];r['fit_error']=r['es']-r['sample_es'];r['total_error']=r['es']-r['truth']
  assert abs(r['fit_error']+r['sample_error']-r['total_error'])<1e-10
 records.append((unit,ev))
P.mkdir(exist_ok=True)
for unit,ev in records:(P/f'heston_{unit}.json').write_text(json.dumps(ev,indent=2))
manifest={'complete':True,'fits':10,'new_parameter_fits':0,'conditional_paths_per_fit':N,'return_steps_per_day':S,'risk_method':'Gaussian-mixture quantile and analytic truncated-normal ES','files':{f'heston_{unit}.json':hashlib.sha256((P/f'heston_{unit}.json').read_bytes()).hexdigest() for unit in range(10)}}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
