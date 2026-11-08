"""Validate completed expanded records, sample sizes, chronology and inference."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from models import load_tree
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'expansion';R=BASE/'results';O=BASE/'analysis';checks=[]
def check(name,condition,detail):
 checks.append(dict(check=name,passed=bool(condition),detail=detail))
 if not condition:raise AssertionError((name,detail))
def main():
 allrecs=[]
 for study,n in [('t',100),('metat',110),('heston',90),('crsp5',200),('crsp100',100),('crsp5cov',100)]:
  recs=[json.loads(p.read_text()) for p in R.glob(study+'_*.json')];recs=[x for x in recs if x.get('study')==study];allrecs+=recs
  check(study+' complete evaluation accounting',len(recs)==n,len(recs))
  if study in ['t','metat','heston']:
   check(study+' independent datasets and scenario counts',{x['dataset'] for x in recs}==set(range(10)) and all(x['n_scenarios']==1000000 for x in recs),dict(datasets=10,scenarios=1000000))
  else:
   ok=all(pd.Timestamp(x['train_end'])<pd.Timestamp(x['validation_start'])<=pd.Timestamp(x['validation_end'])<pd.Timestamp(x['test_start'])<=pd.Timestamp(x['test_end']) for x in recs)
   check(study+' chronological selection',ok,'training precedes validation, validation precedes test')
 nn=[x for x in allrecs if x['model'].startswith(('Flow','VAE','WGAN'))]
 check('Fresh neural fit count',len(nn)==500,len(nn))
 selected=all(x['info']['validation_step']==min(x['info']['validation_trace'],key=lambda z:z[1])[0] and x['info']['validation_scenarios']==20000 for x in nn)
 check('All checkpoints minimize their saved validation FZ0 trace',selected,len(nn))
 sizes=all(len(x['info']['validation_trace'])==(16 if x['model'].startswith('VAE') else 12) for x in nn)
 check('Unchanged complete training budgets',sizes,'12 flow/GAN and 16 VAE validation checkpoints')
 for panel,seedcount in [('crsp5',2),('crsp100',1)]:
  a=pd.read_csv(O/(panel+'_forecasts.csv.gz'));n=2514*3*2*(10*seedcount+3)
  check(panel+' unique balanced forecast panel',len(a)==n and not a.duplicated(['date','model','portfolio','alpha','seed']).any(),len(a))
  q=a['var'].to_numpy();e=a.es.to_numpy();loss=a.loss.to_numpy();al=a.alpha.to_numpy()
  reconstructed=np.log(e)+q/e-1+np.where(loss>q,(loss-q)/(al*e),0)
  check(panel+' independent FZ0 reconstruction',np.max(abs(reconstructed-a.fz0))<1e-10,float(np.max(abs(reconstructed-a.fz0))))
  check(panel+' VaR/ES order and integer hits',np.all(e>=q)&np.all(e>0)&np.all(a.hit.to_numpy()==(loss>q).astype(int)),float((e-q).min()))
  inf=pd.read_csv(O/(panel+'_inference.csv'));rej=(inf.low>0)|(inf.high<0);away=abs(inf.p-.05)>.001
  check(panel+' basic-interval/test coherence',np.all(rej[away]==(inf.p[away]<.05)),int(rej.sum()))
  check(panel+' bootstrap probabilities',inf.p.between(0,1).all() and inf.p_holm.between(0,1).all() and np.all(inf.p_holm>=inf.p-1e-12),len(inf))
 dec=pd.read_csv(O/'decomposition.csv');err=abs(dec.fit_mse+dec.sampling_mse+dec['cross']-dec.total_mse)/(1+dec.total_mse)
 check('Expanded MSE identity with cross term',err.max()<1e-11,float(err.max()))
 check('Paired SE uses ten independent datasets',dec.n_datasets.eq(10).all(),len(dec))
 raw=pd.read_csv(O/'synthetic_raw.csv');spread=raw.groupby(['study','dataset','portfolio','alpha']).sample_es.agg(lambda z:z.max()-z.min())
 check('Models share the exact empirical-training baseline',spread.max()<1e-10,float(spread.max()))
 minimum=1e9;maxdiag=0.;mineig=1e9;n=0
 for path in (BASE/'checkpoints').glob('*.meta.json'):
  z=json.loads(path.read_text());m=np.array(z['marg']);P=np.array(z['P']);n+=1
  minimum=min(minimum,m[:,0].min());assert np.all((m[:,0]>=1.05)&(m[:,0]<=200)) and np.all(m[:,2]>0)
  maxdiag=max(maxdiag,float(abs(np.diag(P)-1).max()));mineig=min(mineig,float(np.linalg.eigvalsh(P).min()))
 check('New meta-t bounds and correlation matrices',minimum>=1.05 and maxdiag<1e-11 and mineig>0,dict(fits=n,min_nu=minimum,max_diagonal_error=maxdiag,min_eigenvalue=mineig))
 pars=pd.concat([pd.read_csv(p) for p in R.glob('crsp100_*_dynamic_parameters.csv')],ignore_index=True)
 check('Thirty new high-dimensional portfolio GARCH fits',len(pars)==30 and pars.success.all() and (pars.evt_xi<1).all() and (pars.arch+pars.beta<.9999).all(),len(pars))
 refs=[]
 for law in ['t','metat','heston']:
  a=json.loads((BASE/'data'/f'{law}_reference.json').read_text());refs.extend(a.values())
 check('Four-million-draw reference evaluations',all(x['n']==4000000 and x['tail_draws'] in [4000,40000] for x in refs),len(refs))
 check('Analytic Student ES reference',abs(json.loads((BASE/'data/t_reference.json').read_text())['equal_0.01']['es']-6.06484695)<1e-8,'6.06484695')
 identity=json.loads((O/'filter_identity_check.json').read_text());check('Full-covariance FHS independently reproduces benchmark',identity['passed'] and identity['rows']==2514*3*2,identity)
 source=json.loads((BASE/'data/source_reconstruction_checks.json').read_text());check('Independent CRSP100 source reconstruction',len(source)==10 and sum(x['test_dates'] for x in source)==2514 and all(x['max_export_error']<1e-9 and hashlib.file_digest((BASE/'data'/f"crsp100_{x['year']}.csv").open('rb'),'sha256').hexdigest()==x['sha256'] for x in source),sum(x['test_dates'] for x in source))
 warm=json.loads((BASE/'data/warmup_audit.json').read_text());check('No high-dimensional cold-start scale in retained training',len(warm)==10 and min(x['min_training_scale'] for x in warm)>.2 and max(x['max_asset_innovation_sd'] for x in warm)<1.5,'All ten folds independently inspected')
 checkpoint_files=list((BASE/'checkpoints').glob('*.npz'));finite=True
 for p in checkpoint_files:
  with np.load(p) as z:finite=finite and all(np.isfinite(z[k]).all() for k in z.files)
 check('All new neural checkpoint arrays finite',finite and len(checkpoint_files)==500,len(checkpoint_files))
 (BASE/'checks_outputs.json').write_text(json.dumps(checks,indent=2));print(len(checks),'expanded output checks passed')
if __name__=='__main__':main()
