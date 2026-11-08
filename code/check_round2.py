"""Independent contracts for matched replay, filters, selection and reporting."""
from pathlib import Path
import json
import numpy as np,pandas as pd,jax
from models import Neural,MetaT
from round2_selection import cached_sampler
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'expansion';N=ROOT/'referee_round2';O=N/'analysis';checks=[]
def check(name,passed,detail):
 checks.append(dict(check=name,passed=bool(passed),detail=detail))
 if not passed:raise AssertionError((name,detail))
def main():
 adds=[json.loads(p.read_text()) for p in (N/'results').glob('heston_*.json')]
 check('Complete Heston model additions',len(adds)==30 and all(sum(r['model']==m for r in adds)==10 for m in ['Meta-t','VAE-t','WGAN-i']),len(adds))
 check('Added primary evaluations use one million scenarios',all(r['n_scenarios']==1000000 and r['n_train']==4000 and r['n_val']==1000 for r in adds),len(adds))
 records=[]
 for study,count in [('t',70),('metat',80),('heston',90),('crsp5',70),('crsp100',70)]:
  rs=[json.loads(p.read_text()) for p in (N/'selection').glob(study+'_*.json') if '_step' not in p.name and '_choices' not in p.name];records+=rs
  check(study+' matched trajectories complete',len(rs)==count,len(rs))
 check('Replayed primary parameter equality',all(r['replay_max_parameter_error']<2e-5 for r in records),max(r['replay_max_parameter_error'] for r in records))
 # Independently compare all selected baseline array trees with the retained original states.
 err=0.
 for r in records:
  tag=f"{r['study']}_{r['unit']}_0_{r['model']}";base=N if r['study']=='heston' and r['model'] in ['VAE-t','WGAN-i'] else OLD
  a=Neural.load(base/'checkpoints'/(tag+'.npz'));b=Neural.load(N/'selection'/f"{tag}_step{r['primary_step']}.npz")
  err=max(err,max(float(np.max(abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(jax.tree_util.tree_leaves(a.p),jax.tree_util.tree_leaves(b.p))))
 check('Independent baseline checkpoint-array comparison',err<2e-5,err)
 ok=True;criterion_count=0
 for r in records:
  expected={'fz1','fz2p5','fz5','pooled1','final'}|({'density'} if not r['model'].startswith('WGAN') else set());ok &= set(r['choices'])==expected;criterion_count+=len(r['choices'])
  for k,v in r['choices'].items():
   step=max(x['step'] for x in r['traces']) if k=='final' else min((x for x in r['traces'] if np.isfinite(x[k])),key=lambda x:x[k])['step']
   ok &= step==v['step'] and step==r['evaluations'][k]['step']
 check('Every competing selection uses its recorded validation minimum or final step',ok and criterion_count==2150,criterion_count)
 ch=pd.read_csv(O/'checkpoint_changes.csv');check('Checkpoint changes are paired to the same trajectory',len(ch)==2150 and set(ch.unit[ch.study.str.startswith('crsp')])==set(range(2016,2026)),len(ch))
 for panel in ['crsp5','crsp100']:
  pred=pd.read_csv(O/(panel+'_forecasts.csv.gz'));expected=2514*3*2*(25 if panel=='crsp5' else 15)
  check(panel+' balanced augmented forecasts',len(pred)==expected and not pred.duplicated(['date','model','portfolio','alpha','seed']).any(),len(pred))
  a=pred.alpha.to_numpy();e=pred.es.to_numpy();q=pred['var'].to_numpy();l=pred.loss.to_numpy();s=np.log(e)+q/e-1+np.maximum(l-q,0)/(a*e)
  check(panel+' independent FZ0 and positive-scale ES accounting',np.max(abs(s-pred.fz0))<1e-10 and np.all(e>=q) and np.all(e>0),float(np.max(abs(s-pred.fz0))))
  for c in [.01,100.]:
   sc=np.log(c*e)+(c*q)/(c*e)-1+np.maximum(c*l-c*q,0)/(a*c*e);assert np.max(abs(sc-s-np.log(c)))<1e-10
  inf=pd.read_csv(O/(panel+'_inference.csv'));away=abs(inf.p-.05)>.001
  check(panel+' test/interval coherence with fourteen contrasts',len(inf)==30 and np.all(((inf.low>0)|(inf.high<0))[away]==(inf.p<.05)[away]) and (inf.p_holm>=inf.p-1e-12).all(),len(inf))
  daily=pd.read_csv(O/(panel+'_selection_daily.csv.gz'),index_col=0);check(panel+' common-date selection comparison',daily.shape==(2514,41) and daily.notna().all().all(),list(daily.shape))
 check('FZ0 common-unit shifts cancel in differences',True,'c=0.01 and c=100 verified on both full forecast panels')
 f=json.loads((O/'filter_state_checks.json').read_text());check('GARCH-FHS reuses the original GARCH states',len(f)==120 and max(x['garch_var_max_error'] for x in f)<1e-8,max(x['garch_var_max_error'] for x in f))
 f=pd.read_csv(O/'filter_decomposition.csv');check('Filter/location/tail decomposition is exact date by date',len(f)==12 and f.identity_max_error.max()<1e-10,float(f.identity_max_error.max()))
 syn=pd.read_csv(O/'synthetic_summary.csv');check('Completed synthetic MSE decomposition',len(syn)==66 and np.max(abs(syn.fit_mse+syn.sampling_mse+syn['cross']-syn.total_mse))<1e-10,len(syn))
 for ds in range(10):
  z=MetaT.load(N/'checkpoints'/f'heston_{ds}_0_Meta-t.meta.json');assert np.max(abs(np.diag(z.P)-1))<1e-10 and np.linalg.eigvalsh(z.P).min()>0 and np.all((z.marg[:,0]>=1.05)&(z.marg[:,0]<=200))
 check('Heston meta-t fit has constrained margins and a correlation matrix',True,10)
 lv=json.loads((O/'leverage.json').read_text());check('Paired leverage experiment size and fixed price correlation',lv['paths']==1000000 and lv['batches']==20 and lv['rows'][0]['price_shock_correlation']==lv['rows'][1]['price_shock_correlation']==.549,len(lv['rows']))
 d=list((N/'diagnostics').glob('heston_*.json'));check('Added Heston dependence controls complete',len(d)==30,len(d))
 n=Neural.load(N/'checkpoints/heston_0_0_WGAN-i.npz');a=n.sample(10000,123);b=cached_sampler(n)(10000,123);check('Cached evaluation preserves draws exactly',np.array_equal(a,b),float(np.max(abs(a-b))))
 (O/'round2_checks.json').write_text(json.dumps(checks,indent=2));print(len(checks),'round-two checks passed')
if __name__=='__main__':main()
