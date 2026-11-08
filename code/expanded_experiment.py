"""Fresh common-FZ0 fits and large-scenario evaluations. No archived state reuse.

Synthetic: 10 independent estimation datasets, one optimizer seed each;
20,000 validation scenarios at alpha=.01; 1,000,000 evaluation scenarios.
CRSP: annual estimation/validation/test split, 100,000 evaluation scenarios.
Only validation observations select checkpoints. Protocol is frozen separately.
"""
import argparse,json,time,gc,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import jax
from models import Neural,MetaT
from experiment import sample_law,heston,es,ewma
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'expansion'
for f in ['results','checkpoints','data','logs']:(OUT/f).mkdir(exist_ok=True,parents=True)
MAPPING={'Flow-G':('flow','gaussian'),'Flow-t':('flow','t-elliptical'),'Flow-L':('flow','laplace'),'VAE-G':('vae','gaussian'),'VAE-t':('vae','t-elliptical'),'VAE-H':('vae-scale','gaussian'),'WGAN-G':('gan','gaussian'),'WGAN-t':('gan','t-elliptical'),'WGAN-i':('gan','t-independent')}
COMMON=['Historical','Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']

def score(loss,q,e,a=.01):
 return np.log(e)+q/e-1+np.maximum(loss-q,0)/(a*e)

def risk_pairs(samples,projections,alpha=.01):
 # Sample columns projected to forecast losses. Fractional empirical ES is exact.
 loss=-np.asarray(samples)@np.asarray(projections).T
 n=len(loss);m=n*alpha;k=int(np.floor(m));frac=m-k;cut=n-k
 part=np.partition(loss,[cut-1,min(cut,n-1)],axis=0)
 q=part[cut-1];e=(part[cut:].sum(0)+frac*part[cut-1])/m
 return q,e

class FZValidation:
 def __init__(self,returns,projections,transform=None):
  self.actual=np.asarray(returns);self.proj=np.atleast_2d(projections);self.transform=transform
 def __call__(self,samples):
  if self.transform is not None:samples=self.transform(samples)
  if len(self.proj)==1:
   q,e=risk_pairs(samples,self.proj)
   return float(np.mean(score(self.actual,q[0],e[0]))) if e[0]>0 and np.isfinite(e[0]) else np.inf
  total=0.;n=0
  for start in range(0,len(self.proj),32):
   p=self.proj[start:start+32];q,e=risk_pairs(samples,p)
   if not np.all(np.isfinite(e)&(e>0)):return np.inf
   s=score(self.actual[start:start+32],q,e);total+=s.sum();n+=len(s)
  return float(total/n)

class Fitted:
 def __init__(self,name,X,V,T,seed,path,validator):
  self.name=name;self.info={};self.X=X;self.path=path;self.n=None;self.mt=None
  if name=='Historical':self.lp=None
  elif name=='Gaussian':
   self.mu=X.mean(0);self.cov=np.cov(X,rowvar=False)+np.eye(X.shape[1])*1e-7;self.lp=st.multivariate_normal.logpdf(T,self.mu,self.cov)
  elif name in ['Meta-t','Flow-meta-t']:
   mp=path.with_suffix('.meta.json')
   self.mt=MetaT.load(mp) if mp.exists() else MetaT().fit(X)
   self.mt.save(mp);self.info={'marginal_df':self.mt.marg[:,0].tolist(),'copula_df':self.mt.nu}
   if name=='Meta-t':self.lp=self.mt.logpdf(T)
   else:
    if path.exists():self.n=Neural.load(path)
    else:
     v=FZValidation(validator.actual,validator.proj,transform=self.mt.from_y)
     self.n=Neural('flow',X.shape[1],seed,base='t-elliptical',fixed_nu=self.mt.nu,P=self.mt.P).fit(self.mt.to_y(X),self.mt.to_y(V),path,risk_validation=v)
    yt=self.mt.to_y(T);self.lp=self.n.logpdf(yt)+self.mt.logjac(T,yt)
  else:
   kind,base=MAPPING[name]
   self.n=Neural.load(path) if path.exists() else Neural(kind,X.shape[1],seed,base=base).fit(X,V,path,risk_validation=validator)
   self.lp=self.n.logpdf(T)
  if self.n is not None:
   self.info.update(validation_step=self.n.best_step,validation_fz0=min(x[1] for x in self.n.trace),validation_trace=self.n.trace,fit_seconds=self.n.fit_seconds,validation_scenarios=20000)
   if name=='Flow-t':self.info['base_df']=float(self.n.nu(self.n.p))
 def sample(self,n,seed):
  if self.name=='Historical':return self.X[np.random.default_rng(seed).integers(len(self.X),size=n)]
  if self.name=='Gaussian':return np.random.default_rng(seed).multivariate_normal(self.mu,self.cov,n)
  if self.name=='Meta-t':return self.mt.sample(n,seed)
  x=self.n.sample(n,seed)
  return self.mt.from_y(x) if self.name=='Flow-meta-t' else x

def law_weights(study):
 if study=='heston':return {'equal':np.ones(25)/5}
 return {'equal':np.ones(2)/2,'first':np.array([1.,0.]),'second':np.array([0.,1.])}

def references():
 for law in ['t','metat','heston']:
  p=OUT/'data'/f'{law}_reference.json'
  if p.exists():continue
  w=law_weights(law);loss={pn:[] for pn in w}
  for b in range(40):
   x=heston(100000,700000+b) if law=='heston' else sample_law(law,100000,700000+b)
   for pn,v in w.items():loss[pn].append(-x@v)
   if b%10==0:print('Reference',law,b,flush=True)
  result={};arrays={}
  for pn,parts in loss.items():
   values=np.concatenate(parts);arrays[pn]=values
   for a in [.01,.001]:
    val=es(values,a)
    if law=='t':
     v=w[pn];scale=np.sqrt(v@np.array([[1,.5],[.5,1]])@v);q=st.t.isf(a,3);val=float(scale*(3+q*q)/2*st.t.pdf(q,3)/a)
    result[f'{pn}_{a}']=dict(es=val,mc_es=es(values,a),mc_se=float(np.std([es(z,a) for z in parts],ddof=1)/np.sqrt(len(parts))),n=4000000,tail_draws=int(4000000*a),analytic=law=='t')
  np.savez_compressed(OUT/'data'/f'{law}_reference_losses.npz',**arrays)
  p.write_text(json.dumps(result,indent=2))

def synthetic(law,datasets,names=None):
 f=lambda n,seed:heston(n,seed) if law=='heston' else sample_law(law,n,seed)
 w=law_weights(law);truth=json.loads((OUT/'data'/f'{law}_reference.json').read_text());T=f(20000,999)
 default=['Historical','Gaussian','Meta-t','Flow-G','Flow-t','VAE-G','VAE-t','WGAN-G','WGAN-t','WGAN-i']
 if law=='metat':default.append('Flow-meta-t')
 if law=='heston':default=['Historical','Gaussian','Flow-G','Flow-t','Flow-L','VAE-G','VAE-H','WGAN-G','WGAN-t']
 names=names or default
 for ds in datasets:
  # New seeds do not overlap original three estimation datasets or reference.
  X=f(4000,50000+ds);V=f(1000,60000+ds);validator=FZValidation(-V@w['equal'],w['equal'])
  for name in names:
   tag=f'{law}_{ds}_0_{name}';p=OUT/'results'/f'{tag}.json'
   if p.exists():continue
   start=time.time();model=Fitted(name,X,V,T,80000+ds,OUT/'checkpoints'/f'{tag}.npz',validator)
   losses={pn:[] for pn in w}
   for b in range(10):
    samples=model.sample(100000,900000+ds*100+b)
    if not np.isfinite(samples).all():raise ValueError(tag+' nonfinite scenarios')
    for pn,v in w.items():losses[pn].append(-samples@v)
   metrics=[]
   for pn,parts in losses.items():
    loss=np.concatenate(parts)
    for a in [.01,.001]:
     estimate=es(loss,a);target=truth[f'{pn}_{a}']['es'];emp=es(-X@w[pn],a)
     metrics.append(dict(portfolio=pn,alpha=a,var=float(np.quantile(loss,1-a,method='inverted_cdf')),es=estimate,truth=target,sample_es=emp,fit_error=estimate-emp,sample_error=emp-target,total_error=estimate-target,scenario_mc_se=float(np.std([es(z,a) for z in parts],ddof=1)/np.sqrt(len(parts))),reference_mc_se=truth[f'{pn}_{a}']['mc_se']))
   rec=dict(study=law,dataset=ds,seed=0,model=name,n_scenarios=1000000,n_train=4000,n_val=1000,n_test=20000,nll=None if model.lp is None else float(-model.lp.mean()),nll_kind='IWAE64' if name.startswith('VAE') else 'exact' if model.lp is not None else 'none',info=model.info,metrics=metrics,seconds=time.time()-start,checkpoint_selection='mean validation FZ0, equal portfolio, alpha=.01')
   p.write_text(json.dumps(rec,indent=2));print(tag,'ES error',metrics[0]['total_error'],'sec',round(rec['seconds'],1),flush=True)
   del model,losses,samples;gc.collect();jax.clear_caches()

def forecast(samples,scales,returns,dates,name,seed,year,weights):
 rows=[]
 for pn,w in weights.items():
  for start in range(0,len(scales),32):
   block=scales[start:start+32]
   projections=np.einsum('tij,i->tj',block,w) if block.ndim==3 else block*w
   actual=-returns[start:start+32]@w
   for alpha in [.01,.025]:
    q,e=risk_pairs(samples,projections,alpha)
    if np.any(e<=0):raise ValueError((name,year,'nonpositive ES'))
    s=score(actual,q,e,alpha);res=e-q-np.maximum(actual-q,0)/alpha
    for i in range(len(actual)):
     rows.append(dict(date=str(dates[start+i].date()),fold=year,model=name,seed=seed,portfolio=pn,alpha=alpha,loss=float(actual[i]),var=float(q[i]),es=float(e[i]),hit=int(actual[i]>q[i]),fz0=float(s[i]),es_residual=float(res[i])))
 return pd.DataFrame(rows)

def crsp(panel,years,seeds,names=None):
 names=names or COMMON
 for year in years:
  path=ROOT/'data/crsp_size_portfolios.csv' if panel in ['crsp5','crsp5cov'] else OUT/'data'/f'crsp100_{year}.csv'
  f=pd.read_csv(path,index_col=0,parse_dates=True);X=f.values*100;d=X.shape[1]
  if panel=='crsp5cov':
   cov=np.cov(X[:252],rowvar=False);scales=[];Z=[]
   for xt in X:
    C=np.linalg.cholesky(cov+1e-10*np.eye(d));scales.append(C);Z.append(np.linalg.solve(C,xt));cov=.94*cov+.06*np.outer(xt,xt)
   scales=np.asarray(scales);Z=np.asarray(Z)
  else:scales=ewma(X);Z=X/scales
  train=(f.index.year>=max(2006,year-11))&(f.index.year<=year-2);val=f.index.year==year-1;test=f.index.year==year
  assert f.index[train].max()<f.index[val].min()<f.index[test].min()
  weights={'equal':np.ones(d)/d,'small':np.eye(d)[0],'large':np.eye(d)[-1]} if panel in ['crsp5','crsp5cov'] else {'equal':np.ones(d)/d,'top10':np.r_[np.ones(10)/10,np.zeros(d-10)],'largest':np.eye(d)[0]}
  w=weights['equal'];projections=np.einsum('tij,i->tj',scales[val],w) if panel=='crsp5cov' else scales[val]*w
  validator=FZValidation(-X[val]@w,projections)
  for seed in seeds:
   for name in names:
    tag=f'{panel}_{year}_{seed}_{name}';p=OUT/'results'/f'{tag}.json';csv=p.with_suffix('.csv.gz')
    if p.exists() and csv.exists():continue
    start=time.time();model=Fitted(name,Z[train],Z[val],Z[test],180000+10*year+seed,OUT/'checkpoints'/f'{tag}.npz',validator)
    samples=Z[train] if name=='Historical' else model.sample(100000,190000+10*year+seed)
    pred=forecast(samples,scales[test],X[test],f.index[test],name,seed,year,weights);pred.to_csv(csv,index=False)
    logdet=np.log(np.diagonal(scales[test],axis1=1,axis2=2)).sum(1) if panel=='crsp5cov' else np.log(scales[test]).sum(1)
    rec=dict(study=panel,model=name,fold=year,seed=seed,dimension=d,n_train=int(train.sum()),n_val=int(val.sum()),n_test=int(test.sum()),train_start=str(f.index[train].min().date()),train_end=str(f.index[train].max().date()),validation_start=str(f.index[val].min().date()),validation_end=str(f.index[val].max().date()),test_start=str(f.index[test].min().date()),test_end=str(f.index[test].max().date()),n_scenarios=len(samples),info=model.info,nll=None if model.lp is None else float(np.mean(-model.lp+logdet)),seconds=time.time()-start,checkpoint_selection='mean validation FZ0, equal portfolio, alpha=.01')
    p.write_text(json.dumps(rec,indent=2));print(tag,'FZ',pred[(pred.portfolio=='equal')&(pred.alpha==.01)].fz0.mean(),'sec',round(rec['seconds'],1),flush=True)
    del model,samples,pred;gc.collect();jax.clear_caches()

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',required=True,choices=['references','t','metat','heston','crsp5','crsp5cov','crsp100']);p.add_argument('--datasets',nargs='+',type=int,default=list(range(10)));p.add_argument('--years',nargs='+',type=int,default=list(range(2016,2026)));p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--models',nargs='+');a=p.parse_args()
 if a.study=='references':references()
 elif a.study in ['t','metat','heston']:synthetic(a.study,a.datasets,a.models)
 else:crsp(a.study,a.years,a.seeds,a.models)
if __name__=='__main__':main()
