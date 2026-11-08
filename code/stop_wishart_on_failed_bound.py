"""Freeze a decisive numerical rejection; no performance-based model selection."""
import json,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd
from analyze_wishart_completion import ensemble
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'wishart_completion';OUT=W/'early_stop';OUT.mkdir(exist_ok=True)
D={};inputs=[]
for n,s in [(1024,8),(2048,8),(1024,16)]:
 frames=[]
 for b in [0,1]:
  tag=f'y2020_b{b}_n{n}_s{s}_p8192_w252_m3_anti_lag2_moment_ap25_pathgauge';p=W/'checkpoints'/(tag+'.json');data=p.read_bytes();x=json.loads(data);dest=OUT/(tag+'.json')
  if dest.exists():x=json.loads(dest.read_text());data=dest.read_bytes()
  else:dest.write_bytes(data)
  f=pd.DataFrame(x['rows']);assert not f.duplicated(['date','portfolio','alpha']).any();frames.append(f)
  inputs.append(dict(tag=tag,snapshot=str(dest.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),next_index=x['next_index'],forecast_dates=f.date.nunique()))
 dates=set(frames[0].date)&set(frames[1].date);raw=pd.concat([f[f.date.isin(dates)] for f in frames]);en=ensemble(raw);D[n,s]=en;en.to_csv(OUT/f'ensemble_n{n}_s{s}.csv.gz',index=False)
rows=[]
for label,cfg in [('Particles',(2048,8)),('Time grid',(1024,16))]:
 p=D[1024,8].merge(D[cfg],on=['date','fold','portfolio','alpha'],suffixes=('_coarse','_fine'),validate='one_to_one')
 for (pn,a),f in p.groupby(['portfolio','alpha']):
  squared=((f.es_fine-f.es_coarse)/f.es_fine)**2;lower=float(np.sqrt(squared.sum()/253));rms=float(np.sqrt(squared.mean()))
  rows.append(dict(refinement=label,portfolio=pn,alpha=a,dates=len(f),first_date=f.date.min(),last_date=f.date.max(),prefix_delta_fz0=float((f.fz0_fine-f.fz0_coarse).mean()),prefix_es_relative_rms=rms,full_year_es_rms_lower_bound=lower,annual_dates=253,tolerance=.02,decisively_fails=bool(lower>.02)))
rows=pd.DataFrame(rows);primary=rows[(rows.portfolio=='equal')&(rows.alpha==.01)]
assert primary.decisively_fails.any(),'No decisive lower-bound failure; do not stop runs'
rows.to_csv(OUT/'refinement_bounds.csv',index=False)
out=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='The squared discrepancies already observed exceed the entire annual numerical-error budget; uncomputed nonnegative terms cannot reverse this failure.',rule='sqrt(sum_observed ((ES_fine-ES_coarse)/ES_fine)^2 / 253) > .02',stopping_basis='pre-existing numerical tolerance; no selection by favorable FZ0',full_year_completed=False,primary_gate_passed=False,inputs=inputs,primary_bounds=primary.to_dict('records'))
(OUT/'decision.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
