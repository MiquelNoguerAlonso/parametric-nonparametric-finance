"""Ten-dataset radial repair and Heston dependence checks on retained fits.

These one-percent/dependence diagnostics use 100,000 scenarios per fit; the
primary 1% and 0.1% risk results retain their separate million-scenario draws.
"""
import argparse,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import jax
from models import Neural,MetaT
from experiment import sample_law,heston,radial,stylised,es
from analyze import tex_table,fmt
from revision_analysis import label
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'expansion';R=BASE/'results';O=BASE/'analysis';D=BASE/'diagnostics';D.mkdir(exist_ok=True)
def sample(name,X,tag,seed):
 r=np.random.default_rng(seed);path=BASE/'checkpoints'/(tag+'.npz')
 if name=='Historical':return X[r.integers(len(X),size=100000)]
 if name=='Gaussian':return r.multivariate_normal(X.mean(0),np.cov(X,rowvar=False)+np.eye(X.shape[1])*1e-7,100000)
 if name=='Meta-t':return MetaT.load(path.with_suffix('.meta.json')).sample(100000,seed)
 n=Neural.load(path);z=n.sample(100000,seed)
 return MetaT.load(path.with_suffix('.meta.json')).from_y(z) if name=='Flow-meta-t' else z

def compute(law):
 records=[json.loads(p.read_text()) for p in R.glob(law+'_*.json')];records=[x for x in records if x.get('study')==law]
 assert len(records)=={'t':100,'metat':110,'heston':90}[law]
 refs=json.loads((BASE/'data'/f'{law}_reference.json').read_text());truth=np.sort(np.load(BASE/'data'/f'{law}_reference_losses.npz')['equal']);ref_emp=es(truth,.01);xcache={}
 for r in sorted(records,key=lambda x:(x['dataset'],x['model'])):
  ds=r['dataset'];name=r['model'];tag=f'{law}_{ds}_0_{name}';target=D/(tag+'.json')
  if target.exists():continue
  if ds not in xcache:xcache={ds:heston(4000,50000+ds) if law=='heston' else sample_law(law,4000,50000+ds)}
  X=xcache[ds];z=sample(name,X,tag,510000+ds);w=np.ones(25)/5 if law=='heston' else np.ones(2)/2;loss=-z@w
  # Exact Wasserstein integral for empirical sizes in ratio 40:1.
  w1=float(np.mean(abs(np.repeat(np.sort(loss),40)-truth)));risk=es(loss,.01);ratio=.01*abs(risk-ref_emp)/w1
  out=dict(study=law,dataset=ds,model=name,n_scenarios=100000,es=risk,truth_es=refs['equal_0.01']['es'],w1_empirical=w1,es_transport_ratio=ratio)
  assert ratio<=1+1e-10
  if law=='heston':out['stylised']=stylised(z)
  else:
   out['repairs']=[]
   for q in [.025,.05,.1]:
    repaired,fit=radial(z,X,q);fit['es']=None if repaired is None else es(-repaired@w,.01);out['repairs'].append(fit)
  target.write_text(json.dumps(out,indent=2));print('Diagnostic',tag,flush=True)
  del z;gc.collect();jax.clear_caches()
 if law=='heston':
  p=D/'heston_reference.json'
  if not p.exists():p.write_text(json.dumps(dict(n=100000,**stylised(heston(100000,730000))),indent=2))

def analyze():
 rows=[];rr=[];transport=[]
 for law in ['t','metat','heston']:
  recs=[json.loads(p.read_text()) for p in D.glob(law+'_*.json')];recs=[x for x in recs if x.get('study')==law]
  assert len(recs)=={'t':100,'metat':110,'heston':90}[law]
  for name in sorted({x['model'] for x in recs}):
   g=[x for x in recs if x['model']==name];before=np.array([x['es']-x['truth_es'] for x in g]);transport.append(dict(study=law,model=name,w1=float(np.mean([x['w1_empirical'] for x in g])),ratio_mean=float(np.mean([x['es_transport_ratio'] for x in g])),ratio_max=float(np.max([x['es_transport_ratio'] for x in g]))))
   if law=='heston':continue
   row=[law,'Empirical' if name=='Historical' else label(name),fmt(np.sqrt(np.mean(before**2)))]
   for q in [.025,.05,.1]:
    valid=[];delta=[]
    for x in g:
     a=next(v for v in x['repairs'] if v['q']==q)
     if a['finite_es']:
      e=a['es']-x['truth_es'];valid.append(e);delta.append(e*e-(x['es']-x['truth_es'])**2)
    rmse=float(np.sqrt(np.mean(np.array(valid)**2))) if valid else np.nan
    rr.append(dict(study=law,model=name,threshold=q,valid=len(valid),rmse=rmse,delta_mse=float(np.mean(delta)) if delta else np.nan,paired_se=float(np.std(delta,ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan))
    row.append(fmt(rmse)+(f' ({len(valid)}/10)' if len(valid)<10 else ''))
   if name in ['Historical','Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t','WGAN-G','WGAN-t']:rows.append(row)
 pd.DataFrame(rr).to_csv(O/'radial_diagnostics.csv',index=False);pd.DataFrame(transport).to_csv(O/'transport_diagnostics.csv',index=False)
 (O/'radial_table.tex').write_text(tex_table('Expanded radial-splice sensitivity at one percent. Each entry uses ten independent estimation datasets and 100,000 paired before/after scenarios, with the same four-million-draw or analytic reference as the primary study. Columns give ES RMSE for three radial tail fractions. Parentheses identify the retained finite-ES fits when fewer than ten qualify; invalid infinite-mean GPD fits are not assigned a finite ES. Paired MSE differences and standard errors are supplied in the CSV, including all models and adverse outcomes. These diagnostics are separate from the million-scenario primary evaluations.','tab:radial-expanded',['Law','Model','Before','2.5\\%','5\\%','10\\%'],rows,'llrrrr'))
 h=[json.loads(p.read_text()) for p in D.glob('heston_*.json')];h=[x for x in h if x.get('study')=='heston'];ref=json.loads((D/'heston_reference.json').read_text());keys=['kurtosis','cluster','leverage','cross_vol'];rows=[['Reference']+[fmt(ref[k]) for k in keys]];summ=[]
 for name in sorted({x['model'] for x in h}):
  a=[x for x in h if x['model']==name];r={k:float(np.mean([x['stylised'][k] for x in a])) for k in keys};summ.append(dict(model=name,**r));rows.append(['Empirical' if name=='Historical' else label(name)]+[fmt(r[k]) for k in keys])
 pd.DataFrame(summ).to_csv(O/'heston_dependence.csv',index=False)
 (O/'heston_dependence_table.tex').write_text(tex_table('Expanded Heston dependence diagnostics, ten independent fitted datasets and 100,000 scenarios per fit. The reference also uses 100,000 joint windows. Columns are sample excess kurtosis, adjacent-day absolute-return correlation, return/next-day absolute-return correlation, and same-day cross-asset absolute-return correlation. Sample kurtosis for a polynomial model without a fourth moment does not estimate a finite population fourth moment.','tab:heston-expanded-dependence',['Model','Kurtosis','Cluster','Leverage','Cross-vol'],rows))
 (O/'diagnostic_checks.json').write_text(json.dumps([dict(check='All 300 ancillary model diagnostics completed',passed=True,detail=sum(len(list(D.glob(law+'_*.json')))-(1 if law=='heston' else 0) for law in ['t','metat','heston'])),dict(check='Empirical projected Wasserstein/ES inequality',passed=all(x['ratio_max']<=1+1e-10 for x in transport),detail=max(x['ratio_max'] for x in transport))],indent=2))
 print('Expanded ancillary tables completed')
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',choices=['t','metat','heston','analyze'],required=True);a=p.parse_args()
 if a.part=='analyze':analyze()
 else:compute(a.part)
if __name__=='__main__':main()
