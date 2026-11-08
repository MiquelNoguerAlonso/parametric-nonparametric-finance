"""Retrospective, fixed-budget dense-flow and coordinate-filter controls."""
import argparse,json,time,gc,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy import stats,optimize
import jax
from models import Neural,MetaT
from round3_models import DenseFlow
from round2_selection import dataset
from expanded_experiment import risk_pairs,forecast,score
from experiment import es

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'referee_round3'

def source(panel,year):
    path=ROOT/'data/crsp_size_portfolios.csv' if panel=='crsp5' else ROOT/'expansion/data'/f'crsp100_{year}.csv'
    frame=pd.read_csv(path,index_col=0,parse_dates=True);x=frame.values*100
    tr=(frame.index.year>=max(2006,year-11))&(frame.index.year<=year-2)
    va=frame.index.year==year-1;te=frame.index.year==year
    return frame,x,tr,va,te

def ewma_state(x,lam,initial):
    out=[];h=initial
    for value in x:out.append(h);h=lam*h+(1-lam)*value*value
    return np.asarray(out)

def filters(panel,year):
    dest=OUT/'filters'/f'{panel}_{year}.npz'
    if dest.exists():return
    # Separate process mode: the retained GARCH estimator enables float64.
    from dynamic_benchmarks import fit_garch,unpack,variance
    import jax.numpy as jnp
    frame,x,tr,va,te=source(panel,year);start=np.where(tr)[0][0];stop=np.where(te)[0][-1]+1
    xx=x[start:stop];train=x[tr];gmeans=[];emeans=[];gs=[];escal=[];pars=[]
    for col in range(x.shape[1]):
        p,initial,diag=fit_garch(train[:,col]);mu,omega,a,b,nu=map(float,unpack(p))
        v=np.asarray(variance(p,jnp.asarray(xx[:,col]),initial));gmeans.append(mu);gs.append(np.sqrt(v))
        mean=float(train[:,col].mean());centered=train[:,col]-mean;init=float(np.var(centered[:252]))
        def qml(lam):
            h=ewma_state(centered,lam,init)
            return float(np.mean(np.log(h)+centered*centered/h))
        fit=optimize.minimize_scalar(qml,bounds=(.8,.9995),method='bounded',options={'xatol':1e-7})
        if not fit.success:raise RuntimeError('EWMA QML failure')
        emean=mean;emeans.append(emean);escal.append(np.sqrt(ewma_state(xx[:,col]-emean,float(fit.x),init)))
        pars.append(dict(coordinate=str(frame.columns[col]),mu=mu,omega=omega,arch=a,beta=b,nu=nu,initial=initial,
                         ewma_mean=emean,ewma_lambda=float(fit.x),ewma_initial=init,ewma_objective=float(fit.fun),**diag))
    np.savez_compressed(dest,returns=xx,garch_means=gmeans,garch_scales=np.column_stack(gs),
                        ewma_means=emeans,ewma_scales=np.column_stack(escal),
                        train=tr[start:stop],val=va[start:stop],test=te[start:stop],dates=frame.index[start:stop].values.astype('datetime64[D]').astype(str))
    (OUT/'filters'/f'{panel}_{year}.json').write_text(json.dumps(dict(panel=panel,year=year,parameters=pars,
        training_sha256=hashlib.sha256(train.tobytes()).hexdigest(),selection='training data only',train_end=str(frame.index[tr].max().date())),indent=2))
    print('FILTERS',panel,year,len(pars),flush=True)

class RiskValidation:
    def __init__(self,actual,proj,offset=0.,transform=None):
        self.actual=np.asarray(actual);self.proj=np.atleast_2d(proj);self.offset=offset;self.transform=transform
    def __call__(self,samples):
        if self.transform is not None:samples=self.transform(samples)
        if len(self.proj)==1:
            q,e=risk_pairs(samples,self.proj);q=q+self.offset;e=e+self.offset
            return float(np.mean(score(self.actual,q[0],e[0]))) if np.isfinite(e[0]) and e[0]>0 else np.inf
        total=0.
        for begin in range(0,len(self.proj),32):
            q,e=risk_pairs(samples,self.proj[begin:begin+32]);q=q+self.offset;e=e+self.offset
            if np.any(~np.isfinite(e)|(e<=0)):return np.inf
            total+=score(self.actual[begin:begin+32],q,e).sum()
        return float(total/len(self.actual))

class Choices:
    def __init__(self):self.best={};self.states={};self.trace=[]
    def __call__(self,n,p,step,value,native):
        self.trace.append(dict(step=step,fz1=value,density=native))
        for key,v in [('fz1',value),('density',native)]:
            if v is not None and np.isfinite(v) and (key not in self.best or v<self.best[key]['value']):
                self.best[key]=dict(step=step,value=v);self.states[key]=p
        self.best['final']=dict(step=step,value=value);self.states['final']=p

def shifted_forecast(samples,data,name,unit):
    means=data.get('means',np.zeros(samples.shape[1]));centered=data['returns']-means
    pred=forecast(samples,data['scales'],centered,data['dates'],name,0,unit,{'equal':data['w']})
    offset=-float(means@data['w'])
    pred['loss']+=offset;pred['var']+=offset;pred['es']+=offset
    if (pred.es<=0).any():raise ValueError('Nonpositive ES after physical mean')
    pred['fz0']=score(pred.loss,pred['var'],pred.es,pred.alpha)
    pred['es_residual']=pred.es-pred['var']-np.maximum(pred.loss-pred['var'],0)/pred.alpha
    return pred

def coordinate_data(panel,year,kind='garch'):
    z=np.load(OUT/'filters'/f'{panel}_{year}.npz');x=z['returns'];means=z[kind+'_means'];sc=z[kind+'_scales'];res=(x-means)/sc
    tr,va,te=z['train'],z['val'],z['test'];w=np.ones(x.shape[1])/x.shape[1]
    return dict(X=res[tr],V=res[va],T=res[te],ports={'equal':(-x[va]@w,sc[va]*w)},seed=180000+10*year,
                w=w,scales=sc[te],returns=x[te],dates=pd.to_datetime(z['dates'][te]),means=means,offset=-float(means@w))

def evaluate(model,mt,data,study,unit,tag):
    sample=lambda n,seed:mt.from_y(model.sample(n,seed)) if mt else model.sample(n,seed)
    target=mt.to_y(data['T']) if mt else data['T'];lp=model.logpdf(target)
    if mt:lp=lp+mt.logjac(data['T'],target)
    if study.startswith('crsp'):lp=lp-np.log(data['scales']).sum(1)
    result=dict(nll=float(-np.mean(lp)))
    if study=='heston':
        parts=[-sample(100000,900000+100*unit+b)@data['w'] for b in range(10)]
        loss=np.concatenate(parts);truth=json.loads((ROOT/'expansion/data/heston_reference.json').read_text());metrics=[]
        for alpha in [.01,.001]:
            value=es(loss,alpha);emp=es(-data['X']@data['w'],alpha);ref=truth[f'equal_{alpha}']['es']
            metrics.append(dict(alpha=alpha,es=value,truth=ref,sample_es=emp,total_error=value-ref,fit_error=value-emp,sample_error=emp-ref,
                scenario_mc_se=float(np.std([es(p,alpha) for p in parts],ddof=1)/np.sqrt(10))))
        result['metrics']=metrics
    else:
        pred=shifted_forecast(sample(100000,190000+10*unit),data,tag,unit)
        name=tag+'_forecasts.csv.gz';pred.to_csv(OUT/'results'/name,index=False);result['forecasts']=name
    return result

def neural(study,unit,family,names=None):
    data=dataset(study,unit) if family=='lu' else coordinate_data(study,unit)
    names=names or (['Flow-G','Flow-t'] if family=='lu' else ['Flow-G','Flow-t','Flow-meta-t'])
    for base_name in names:
        tag=f'{family}_{study}_{unit}_{base_name}';out=OUT/'results'/(tag+'.json')
        if out.exists():continue
        start=time.time();mt=None
        if base_name=='Flow-meta-t':mt=MetaT.load(OUT/'checkpoints'/f'coord_{study}_{unit}.meta.json')
        cls=DenseFlow if family=='lu' else Neural;base='gaussian' if base_name=='Flow-G' else 't-elliptical'
        actual,proj=data['ports']['equal'];valid=RiskValidation(actual,proj,data.get('offset',0.),mt.from_y if mt else None)
        path=OUT/'checkpoints'/(tag+'_choices.json')
        if path.exists():metadata=json.loads(path.read_text())
        else:
            obs=Choices();n=cls('flow',data['X'].shape[1],data['seed'],base=base,steps=3000,
                **(dict(P=mt.P,fixed_nu=mt.nu) if mt else {}))
            n.fit(mt.to_y(data['X']) if mt else data['X'],mt.to_y(data['V']) if mt else data['V'],risk_validation=valid,checkpoint_observer=obs)
            if not all(k in obs.best for k in ['fz1','density','final']):raise RuntimeError('Missing selection state')
            for step in sorted({q['step'] for q in obs.best.values()}):
                k=next(k for k in obs.best if obs.best[k]['step']==step);n.p=obs.states[k];n.best_step=step
                n.save(OUT/'checkpoints'/f'{tag}_step{step}.npz')
            metadata=dict(study=study,unit=unit,family=family,model=base_name,choices=obs.best,trace=obs.trace,
                          seed=data['seed'],dimension=data['X'].shape[1],fit_seconds=time.time()-start,
                          training_sha256=hashlib.sha256(data['X'].tobytes()).hexdigest())
            path.write_text(json.dumps(metadata,indent=2));del n,obs;gc.collect();jax.clear_caches()
        evaluated={}
        for rule in (['fz1','density','final'] if family=='lu' else ['fz1']):
            step=metadata['choices'][rule]['step'];ep=OUT/'results'/f'{tag}_step{step}_evaluation.json'
            if ep.exists():ev=json.loads(ep.read_text())
            else:
                model=cls.load(OUT/'checkpoints'/f'{tag}_step{step}.npz')
                ev=evaluate(model,mt,data,study,unit,f'{tag}_step{step}')
                ev['base_df']=float(model.nu(model.p)) if base!='gaussian' else None
                ep.write_text(json.dumps(ev,indent=2));del model;gc.collect();jax.clear_caches()
            evaluated[rule]=dict(step=step,**ev)
        out.write_text(json.dumps(dict(**metadata,evaluations=evaluated,seconds=time.time()-start),indent=2))
        print('NEURAL',tag,round(time.time()-start,1),flush=True)
    del data;gc.collect();jax.clear_caches()

class Gaussian:
    def __init__(self,x):self.mu=x.mean(0);self.cov=np.cov(x,rowvar=False)+1e-7*np.eye(x.shape[1])
    def logpdf(self,x):return stats.multivariate_normal.logpdf(x,self.mu,self.cov)
    def sample(self,n,seed):return np.random.default_rng(seed).multivariate_normal(self.mu,self.cov,n)

def coordinate_baselines(panel,year):
    for kind,names in [('ewma',['FHS']),('garch',['FHS','Gaussian','Meta-t'])]:
        data=coordinate_data(panel,year,kind)
        for name in names:
            tag=f'coord_{kind}_{panel}_{year}_{name}';out=OUT/'results'/(tag+'.json')
            if out.exists():continue
            if name=='FHS':samples=data['X'];lp=None
            elif name=='Gaussian':
                model=Gaussian(data['X']);samples=model.sample(100000,190000+10*year);lp=model.logpdf(data['T'])
            else:
                path=OUT/'checkpoints'/f'coord_{panel}_{year}.meta.json'
                model=MetaT.load(path) if path.exists() else MetaT().fit(data['X']);model.save(path)
                samples=model.sample(100000,190000+10*year);lp=model.logpdf(data['T'])
            pred=shifted_forecast(samples,data,tag,year);csv=tag+'_forecasts.csv.gz';pred.to_csv(OUT/'results'/csv,index=False)
            out.write_text(json.dumps(dict(study=panel,unit=year,family='coord_'+kind,model=name,
                nll=None if lp is None else float(np.mean(-lp+np.log(data['scales']).sum(1))),forecasts=csv,n_train=len(data['X']),n_test=len(data['T'])),indent=2))
            print('BASELINE',tag,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['filters','baselines','lu','coord'],required=True)
    p.add_argument('--study',required=True);p.add_argument('--units',type=int,nargs='+',required=True);p.add_argument('--models',nargs='+');a=p.parse_args()
    for unit in a.units:
        if a.task=='filters':filters(a.study,unit)
        elif a.task=='baselines':coordinate_baselines(a.study,unit)
        else:neural(a.study,unit,a.task,a.models)

if __name__=='__main__':main()
