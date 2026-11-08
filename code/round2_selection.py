"""Matched-trajectory checkpoint sensitivity; no test observation selects a state."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np,pandas as pd,jax
from models import Neural,MetaT,load_tree
from experiment import heston,sample_law,ewma,es
from expanded_experiment import FZValidation,MAPPING,forecast
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'expansion';NEW=ROOT/'referee_round2';S=NEW/'selection'
S.mkdir(exist_ok=True)

def cached_sampler(n):
 fn=jax.jit(lambda k:n.sample_std(n.p,k,10000))
 def draw(count,seed):
  key=jax.random.PRNGKey(seed);parts=[]
  for _ in range((count+9999)//10000):
   key,k=jax.random.split(key);parts.append(np.asarray(fn(k)))
  return np.concatenate(parts)[:count]*n.scale+n.loc
 return draw

def criteria(samples,portfolios):
 result={};pooled=[]
 for pn,(actual,proj) in portfolios.items():
  proj=np.atleast_2d(proj);tot={a:0. for a in ([.01,.025,.05] if pn=='equal' else [.01])};count=0
  for start in range(0,len(proj),32):
   pp=proj[start:start+32];active=np.any(pp!=0,axis=0);loss=-samples[:,active]@pp[:,active].T;n=len(loss)
   cuts=[]
   for a in tot:
    cut=n-int(np.floor(n*a));cuts.extend([cut-1,min(cut,n-1)])
   part=np.partition(loss,sorted(set(cuts)),axis=0)
   for a in tot:
    m=n*a;k=int(np.floor(m));cut=n-k;q=part[cut-1];e=(part[cut:].sum(0)+(m-k)*part[cut-1])/m
    act=actual if len(proj)==1 else actual[start:start+32]
    val=np.inf if np.any(~np.isfinite(e)|(e<=0)) else float(np.sum(np.log(e)+q/e-1+np.maximum(act-q,0)/(a*e)))
    tot[a]+=val
   count+=len(actual) if len(proj)==1 else len(pp)
  means={a:v/count for a,v in tot.items()};pooled.append(means[.01])
  if pn=='equal':result.update(fz1=means[.01],fz2p5=means[.025],fz5=means[.05])
 result['pooled1']=float(np.mean(pooled));return result

class Observer:
 def __init__(self,portfolios,transform=None):self.portfolios=portfolios;self.transform=transform;self.best={};self.states={};self.traces=[];self.draw=None
 def __call__(self,n,p,step,value,native):
  if self.draw is None:self.draw=jax.jit(lambda q:n.sample_std(q,jax.random.PRNGKey(424242),20000))
  z=np.asarray(self.draw(p))*n.scale+n.loc
  if self.transform is not None:z=self.transform(z)
  values=criteria(z,self.portfolios);assert (np.isinf(values['fz1']) and np.isinf(value)) or abs(values['fz1']-value)<1e-6*(1+abs(value)),(values['fz1'],value)
  if native is not None:values['density']=float(native)
  self.traces.append(dict(step=step,**values))
  for k,v in values.items():
   if np.isfinite(v) and (k not in self.best or v<self.best[k]['value']):self.best[k]=dict(step=step,value=v);self.states[k]=p
  self.best['final']=dict(step=step,value=values['fz1']);self.states['final']=p

def dataset(study,unit):
 if study in ['t','metat','heston']:
  f=lambda n,s:heston(n,s) if study=='heston' else sample_law(study,n,s)
  X=f(4000,50000+unit);V=f(1000,60000+unit);T=f(20000,999);d=X.shape[1]
  weights={'equal':np.ones(d)/(5 if d==25 else d)}
  if d==25:
   for i in range(5):weights[f'asset{i+1}']=np.repeat(np.eye(5)[i],5)
  else:weights.update({f'asset{i+1}':np.eye(d)[i] for i in range(d)})
  ports={k:(-V@w,w[None,:]) for k,w in weights.items()}
  return dict(X=X,V=V,T=T,ports=ports,seed=80000+unit,w=weights['equal'])
 path=ROOT/'data/crsp_size_portfolios.csv' if study=='crsp5' else OLD/'data'/f'crsp100_{unit}.csv'
 f=pd.read_csv(path,index_col=0,parse_dates=True);x=f.values*100;sc=ewma(x);z=x/sc;d=x.shape[1]
 tr=(f.index.year>=max(2006,unit-11))&(f.index.year<=unit-2);va=f.index.year==unit-1;te=f.index.year==unit
 weights={'equal':np.ones(d)/d,'small':np.eye(d)[0],'large':np.eye(d)[-1]} if d==5 else {'equal':np.ones(d)/d,'top10':np.r_[np.ones(10)/10,np.zeros(90)],'largest':np.eye(d)[0]}
 return dict(X=z[tr],V=z[va],T=z[te],ports={k:(-x[va]@w,sc[va]*w) for k,w in weights.items()},seed=180000+10*unit,w=weights['equal'],scales=sc[te],returns=x[te],dates=f.index[te])

def primary(study,unit,name):
 tag=f'{study}_{unit}_0_{name}';base=NEW if study=='heston' and name in ['Meta-t','VAE-t','WGAN-i'] else OLD
 p=base/'results'/(tag+'.json');return tag,base,json.loads(p.read_text())

def work(study,units,names=None):
 for unit in units:
  data=dataset(study,unit)
  default=['Flow-G','Flow-t','VAE-G','VAE-t','WGAN-G','WGAN-t','WGAN-i'] if study in ['t','metat'] else ['Flow-G','Flow-t','Flow-L','VAE-G','VAE-H','VAE-t','WGAN-G','WGAN-t','WGAN-i'] if study=='heston' else ['Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']
  if study=='metat':default.append('Flow-meta-t')
  for name in names or default:
   tag,base,ref=primary(study,unit,name);out=S/(tag+'.json')
   if out.exists():continue
   choices=S/(tag+'_choices.json');mt=None
   if name=='Flow-meta-t':mt=MetaT.load(base/'checkpoints'/(tag+'.meta.json'));kind,noise='flow','t-elliptical'
   else:kind,noise=MAPPING[name]
   start=time.time();path=base/'checkpoints'/(tag+'.npz')
   if choices.exists():meta=json.loads(choices.read_text())
   else:
    obs=Observer(data['ports'],mt.from_y if mt else None);actual,proj=data['ports']['equal'];validator=FZValidation(actual,proj,mt.from_y if mt else None)
    n=Neural(kind,data['X'].shape[1],data['seed'],base=noise,**(dict(fixed_nu=mt.nu,P=mt.P) if mt else {}))
    n.fit(mt.to_y(data['X']) if mt else data['X'],mt.to_y(data['V']) if mt else data['V'],risk_validation=validator,checkpoint_observer=obs)
    original=Neural.load(path);aa=jax.tree_util.tree_leaves(original.p);bb=jax.tree_util.tree_leaves(n.p)
    err=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(aa,bb));assert err<2e-5 and n.best_step==original.best_step,(tag,err,n.best_step,original.best_step)
    meta=dict(study=study,unit=unit,model=name,choices=obs.best,traces=obs.traces,replay_max_parameter_error=err,primary_step=original.best_step,training_seconds=time.time()-start)
    saved=set()
    for key,choice in obs.best.items():
     st=choice['step']
     if st in saved:continue
     n.p=obs.states[key];n.best_step=st;n.save(S/f'{tag}_step{st}.npz');saved.add(st)
    choices.write_text(json.dumps(meta,indent=2));del n,obs,original;gc.collect();jax.clear_caches()
   evaluated={}
   for criterion,choice in meta['choices'].items():
    step=choice['step'];ep=S/f'{tag}_step{step}_evaluation.json'
    if ep.exists():ev=json.loads(ep.read_text())
    elif step==meta['primary_step']:
     if study in ['t','metat','heston']:ev=dict(nll=ref['nll'],metrics=[m for m in ref['metrics'] if m['portfolio']=='equal'])
     else:
      a=pd.read_csv(base/'results'/(tag+'.csv.gz'));a=a[a.portfolio.eq('equal')];dest=S/f'{tag}_step{step}_forecasts.csv.gz';a.to_csv(dest,index=False);ev=dict(nll=ref['nll'],forecasts=dest.name)
     ep.write_text(json.dumps(ev,indent=2))
    else:
     n=Neural.load(S/f'{tag}_step{step}.npz');sampler=cached_sampler(n)
     T=mt.to_y(data['T']) if mt else data['T'];lp=n.logpdf(T)
     if lp is not None:
      if mt:lp=lp+mt.logjac(data['T'],T)
      if study.startswith('crsp'):lp=lp-np.log(data['scales']).sum(1)
     ev=dict(nll=None if lp is None else float(-lp.mean()))
     if study in ['t','metat','heston']:
      losses=[]
      for b in range(10):
       z=sampler(100000,900000+unit*100+b);z=mt.from_y(z) if mt else z;losses.append(-z@data['w'])
      loss=np.concatenate(losses);truth=json.loads((OLD/'data'/f'{study}_reference.json').read_text());metrics=[]
      for a in [.01,.001]:
       risk=es(loss,a);target=truth[f'equal_{a}']['es'];sample=es(-data['X']@data['w'],a)
       metrics.append(dict(portfolio='equal',alpha=a,es=risk,truth=target,sample_es=sample,total_error=risk-target,fit_error=risk-sample,sample_error=sample-target))
      ev['metrics']=metrics
     else:
      z=sampler(100000,190000+10*unit);z=mt.from_y(z) if mt else z
      pred=forecast(z,data['scales'],data['returns'],data['dates'],name,0,unit,{'equal':data['w']});dest=S/f'{tag}_step{step}_forecasts.csv.gz';pred.to_csv(dest,index=False);ev['forecasts']=dest.name
     ep.write_text(json.dumps(ev,indent=2));del n;gc.collect();jax.clear_caches()
    evaluated[criterion]=dict(step=step,**ev)
   out.write_text(json.dumps(dict(**meta,evaluations=evaluated,seconds=time.time()-start),indent=2));print('Completed selection',tag,'steps',{k:v['step'] for k,v in meta['choices'].items()},flush=True)
  del data;gc.collect();jax.clear_caches()

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--units',nargs='+',type=int,required=True);p.add_argument('--models',nargs='+');a=p.parse_args();work(a.study,a.units,a.models)
if __name__=='__main__':main()
