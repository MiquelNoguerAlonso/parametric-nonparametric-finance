"""Expand checkpoint candidates without changing archived training trajectories."""
import argparse,copy,gc,json,time
from pathlib import Path
import numpy as np,pandas as pd,jax,jax.numpy as jnp
from models import Neural,MetaT
from round3_models import DenseFlow
from round3_controls import RiskValidation
from round2_selection import cached_sampler
from round4_data import get_data,logdet,joint_event,ROOT,OUT
from round4_parametric import JointStudent
from expanded_experiment import MAPPING,forecast
from experiment import es

STEPS=[0,10,25,50,100,250]

def criteria(samples,portfolios,offset=0.):
    result={};pooled=[]
    for pn,(actual,proj) in portfolios.items():
        proj=np.atleast_2d(proj);tot={a:0. for a in ([.01,.025,.05] if pn=='equal' else [.01])};count=0
        for start in range(0,len(proj),32):
            pp=proj[start:start+32];active=np.any(pp!=0,axis=0);loss=-samples[:,active]@pp[:,active].T;n=len(loss);cuts=[]
            for a in tot:
                cut=n-int(np.floor(n*a));cuts.extend([cut-1,min(cut,n-1)])
            part=np.partition(loss,sorted(set(cuts)),axis=0)
            for a in tot:
                m=n*a;k=int(np.floor(m));cut=n-k;q=part[cut-1]+offset;e=(part[cut:].sum(0)+(m-k)*part[cut-1])/m+offset
                act=actual if len(proj)==1 else actual[start:start+32]
                value=np.inf if np.any(~np.isfinite(e)|(e<=0)) else float(np.sum(np.log(e)+q/e-1+np.maximum(act-q,0)/(a*e)))
                tot[a]+=value
            count+=len(actual) if len(proj)==1 else len(pp)
        vals={a:v/count for a,v in tot.items()};pooled.append(vals[.01])
        if pn=='equal':result.update(fz1=vals[.01],fz2p5=vals[.025],fz5=vals[.05])
    result['pooled1']=float(np.mean(pooled));return result

def names(study,family):
    if family=='lu':return ['Flow-G','Flow-t']
    if family=='coord':return ['Flow-G','Flow-t','Flow-meta-t']
    if study.startswith('crsp'):return ['Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']
    out=['Flow-G','Flow-t','VAE-G','VAE-t','WGAN-G','WGAN-t','WGAN-i']
    if study=='metat':out.append('Flow-meta-t')
    if study=='heston':out=['Flow-G','Flow-t','Flow-L','VAE-G','VAE-H','VAE-t','WGAN-G','WGAN-t','WGAN-i']
    return out

def old_records(study,unit,name,family,seed):
    if family in ['lu','coord']:
        tag=f'{family}_{study}_{unit}_{name}';base=ROOT/'referee_round3';m=json.loads((base/'checkpoints'/(tag+'_choices.json')).read_text())
        choices=copy.deepcopy(m['choices']);trace=m['trace']
        paths={str(c['step']):base/'checkpoints'/f'{tag}_step{c["step"]}.npz' for c in choices.values()}
        mt=base/'checkpoints'/f'coord_{study}_{unit}.meta.json' if name=='Flow-meta-t' else None
    else:
        tag=f'{study}_{unit}_{seed}_{name}';base=ROOT/('referee_round2' if study=='heston' and name in ['VAE-t','WGAN-i'] else 'expansion')
        old=json.loads((base/'results'/(tag+'.json')).read_text());original=base/'checkpoints'/(tag+'.npz')
        selection=ROOT/'referee_round2/selection'/(tag+'.json')
        if seed==0 and selection.exists():
            m=json.loads(selection.read_text());choices=copy.deepcopy(m['choices']);trace=m['traces'];paths={}
            for c in choices.values():
                p=selection.parent/f'{tag}_step{c["step"]}.npz';paths[str(c['step'])]=p if p.exists() else original
        else:
            info=old['info'];choices={'fz1':dict(step=info['validation_step'],value=info['validation_fz0'])}
            trace=[dict(step=s,fz1=v) for s,v in info['validation_trace']];paths={str(info['validation_step']):original}
        mt=base/'checkpoints'/(tag+'.meta.json') if name=='Flow-meta-t' else None
    return tag,choices,trace,paths,mt

def nested_parameters(n,p,data,name,family,study,unit):
    q=copy.deepcopy(p)
    q['layers']=[net[:-1]+[(jnp.zeros_like(net[-1][0]),jnp.zeros_like(net[-1][1]))] for net in p['layers']]
    q['a']=jnp.zeros(n.d);q['b']=jnp.zeros(n.d)
    if family=='lu':
        if name=='Flow-t':
            path=OUT/'checkpoints'/f'original_{study}_{unit}_Student-MLE.json'
            st=JointStudent.load(path);mu=st.mu;cov=st.scatter;v=st.nu-1.05
            q['nu_raw']=jnp.array(v if v>40 else np.log(np.expm1(v)))
        else:mu=data['X'].mean(0);cov=np.cov(data['X'],rowvar=False)+1e-7*np.eye(n.d)
        a=np.linalg.cholesky(cov/(n.scale[:,None]*n.scale[None,:]));diag=np.diag(a)
        q['mix_lower']=jnp.asarray((a/diag[None,:])[np.tril_indices(n.d,-1)])
        q['mix_upper']=jnp.zeros(n.d*(n.d-1)//2);q['mix_logdiag']=jnp.asarray(np.log(diag))
        q['b']=jnp.asarray(np.linalg.solve(a,(mu-n.loc)/n.scale))
    return q

class EarlyObserver:
    def __init__(self,data,mt,name,family,study,unit):
        self.data=data;self.mt=mt;self.name=name;self.family=family;self.study=study;self.unit=unit
        self.best={};self.states={};self.allstates={};self.trace=[];self.draw=None
    def record(self,n,p,step,native):
        if self.draw is None:self.draw=jax.jit(lambda q:n.sample_std(q,jax.random.PRNGKey(424242),20000))
        z=np.asarray(self.draw(p))*n.scale+n.loc
        if self.mt:z=self.mt.from_y(z)
        vals=criteria(z,self.data['ports'],self.data.get('offset',0.))
        if native is not None:vals['density']=float(native)
        self.trace.append(dict(step=step,**vals));self.allstates[step]=p
        for key,val in vals.items():
            # Update 250 is a replay check, already present in the old grid.
            if step<250 and np.isfinite(val) and (key not in self.best or val<self.best[key]['value']):self.best[key]=dict(step=step,value=val);self.states[key]=p
        return vals
    def __call__(self,n,p,step,value,native):
        vals=self.record(n,p,step,native)
        if np.isfinite(value) and abs(vals['fz1']-value)>1e-6*(1+abs(value)):raise RuntimeError('Validation criteria disagree')
        if step==0 and n.kind=='flow':
            q=nested_parameters(n,p,self.data,self.name,self.family,self.study,self.unit)
            v=self.mt.to_y(self.data['V']) if self.mt else self.data['V'];v=jnp.asarray((v-n.loc)/n.scale,dtype=jnp.float32)
            self.record(n,q,-1,float(-n.flow_logpdf(q,v).mean()))

def legacy_evaluation(study,unit,name,family,seed,step):
    if family in ['lu','coord']:
        p=ROOT/'referee_round3/results'/f'{family}_{study}_{unit}_{name}.json'
        if p.exists():
            z=json.loads(p.read_text())
            for ev in z['evaluations'].values():
                if ev['step']==step:
                    ev=copy.deepcopy(ev)
                    if 'forecasts' in ev:ev['forecasts']='referee_round3/results/'+ev['forecasts']
                    return ev
    else:
        oldtag=f'{study}_{unit}_{seed}_{name}';base=ROOT/('referee_round2' if study=='heston' and name in ['VAE-t','WGAN-i'] else 'expansion')
        old=json.loads((base/'results'/(oldtag+'.json')).read_text())
        if old['info']['validation_step']==step:
            ev=dict(nll=old['nll'])
            if 'metrics' in old:ev['metrics']=old['metrics']
            else:ev['forecasts']=str((base/'results'/(oldtag+'.csv.gz')).relative_to(ROOT))
            return ev
        p=ROOT/'referee_round2/selection'/(oldtag+'.json')
        if p.exists():
            for ev in json.loads(p.read_text())['evaluations'].values():
                if ev['step']==step:
                    ev=copy.deepcopy(ev)
                    if 'forecasts' in ev:ev['forecasts']='referee_round2/selection/'+ev['forecasts']
                    return ev
    return None

def evaluate(n,mt,data,study,unit,name,seed,tag,legacy=None):
    sampler=cached_sampler(n)
    draw=lambda count,key:mt.from_y(sampler(count,key)) if mt else sampler(count,key)
    target=mt.to_y(data['T']) if mt else data['T'];lp=n.logpdf(target)
    r={}
    if lp is not None:
        if mt:lp=lp+mt.logjac(data['T'],target)
        nll=-lp+logdet(data);file=OUT/'pointwise'/(tag+'.npz');np.savez_compressed(file,nll=nll)
        r.update(nll=float(nll.mean()),pointwise=str(file.relative_to(ROOT)))
    else:r['nll']=None
    if study.startswith('crsp'):
        z=draw(100000,190000+10*unit+seed);means=data.get('means',np.zeros(n.d))
        if legacy is not None and 'forecasts' in legacy:r['forecasts']=legacy['forecasts']
        else:
            pred=forecast(z,data['scales'],data['returns']-means,data['dates'],name,seed,unit,data['weights'])
            if np.any(means):
                from expanded_experiment import score
                for pn,w in data['weights'].items():
                    mask=pred.portfolio.eq(pn);offset=-float(means@w)
                    for col in ['loss','var','es']:pred.loc[mask,col]+=offset
                pred['fz0']=score(pred.loss,pred['var'],pred.es,pred.alpha);pred['es_residual']=pred.es-pred['var']-np.maximum(pred.loss-pred['var'],0)/pred.alpha
            f=OUT/'results'/(tag+'_forecasts.csv.gz');pred.to_csv(f,index=False);r['forecasts']=str(f.relative_to(ROOT))
        if study!='crsp5cov':r['joint_event']=joint_event(z,data)
    else:
        if legacy is not None and 'metrics' in legacy:
            r['metrics']=legacy['metrics'];r['reused_evaluation']=True;return r
        parts=[-draw(100000,900000+100*unit+b)@data['w'] for b in range(10)];loss=np.concatenate(parts)
        truth=json.loads((ROOT/'expansion/data'/f'{study}_reference.json').read_text());metrics=[]
        for a in [.01,.001]:
            value=es(loss,a);emp=es(-data['X']@data['w'],a);ref=truth[f'equal_{a}']['es']
            metrics.append(dict(portfolio='equal',alpha=a,es=value,truth=ref,sample_es=emp,total_error=value-ref,fit_error=value-emp,sample_error=emp-ref,scenario_mc_se=float(np.std([es(p,a) for p in parts],ddof=1)/np.sqrt(10))))
        r['metrics']=metrics
    return r

def work(study,unit,family='original',seed=0,selected_names=None):
    data=get_data(study,unit,family,seed)
    if family in ['lu','coord']:data['weights']={'equal':data['w']}
    for name in selected_names or names(study,family):
        tag=f'{family}_{study}_{unit}_{seed}_{name}';dest=OUT/'results'/(tag+'.json')
        if dest.exists():continue
        start=time.time();oldtag,oldchoices,oldtrace,paths,mp=old_records(study,unit,name,family,seed)
        mt=MetaT.load(mp) if mp else None;cls=DenseFlow if family=='lu' else Neural
        kind,base=('flow','t-elliptical') if mt else MAPPING[name]
        repaired=OUT/'checkpoints'/(tag+'_choices.json')
        if repaired.exists():meta=json.loads(repaired.read_text())
        else:
            obs=EarlyObserver(data,mt,name,family,study,unit)
            actual,proj=data['ports']['equal'];validator=RiskValidation(actual,proj,data.get('offset',0.),mt.from_y if mt else None)
            n=cls(kind,data['X'].shape[1],data['seed'],base=base,steps=250,**(dict(P=mt.P,fixed_nu=mt.nu) if mt else {}))
            n.fit(mt.to_y(data['X']) if mt else data['X'],mt.to_y(data['V']) if mt else data['V'],risk_validation=validator,checkpoint_observer=obs,checkpoint_steps=STEPS,allow_invalid_prefix=True)
            old250=next(r for r in oldtrace if r['step']==250);new250=next(r for r in obs.trace if r['step']==250);errors={}
            for key in ['fz1','density']:
                if old250.get(key) is None:continue
                a,b=old250[key],new250[key]
                if np.isfinite(a) and np.isfinite(b):
                    errors[key]=abs(a-b)
                    if abs(a-b)>2e-5*(1+abs(a)):raise RuntimeError(('Replay criterion changed',tag,key,a,b))
                elif not (np.isinf(a) and np.isinf(b)):raise RuntimeError('Replay finiteness changed')
            parerr=None
            p250=paths.get('250')
            if p250 and p250.exists():
                saved=cls.load(p250);parerr=max(float(np.max(abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree_util.tree_leaves(n.last_p),jax.tree_util.tree_leaves(saved.p)))
                if parerr>2e-5:raise RuntimeError(('Replay parameters changed',tag,parerr))
                del saved
            choices={};saved_early={}
            for rule,late in oldchoices.items():
                early=obs.best.get(rule);use=rule!='final' and early is not None and early['value']<late['value']
                choice=copy.deepcopy(early if use else late)
                if use:
                    step=choice['step'];path=OUT/'checkpoints'/f'{tag}_step{step}.npz'
                    if step not in saved_early:
                        n.p=obs.allstates[step];n.best_step=step;n.steps=4000 if kind.startswith('vae') else 3000
                        n.save(path);saved_early[step]=path
                else:path=paths[str(choice['step'])]
                choice.update(checkpoint=str(path.relative_to(ROOT)),early_added=use);choices[rule]=choice
            meta=dict(study=study,unit=unit,family=family,seed=seed,model=name,choices=choices,old_choices=oldchoices,early_trace=obs.trace,replay_criterion_errors=errors,replay_parameter_error=parerr,meta_checkpoint=str(mp.relative_to(ROOT)) if mp else None)
            repaired.write_text(json.dumps(meta,indent=2));del obs,n;gc.collect();jax.clear_caches()
        evaluations={};cache={}
        for rule,choice in meta['choices'].items():
            step=choice['step'];key=choice['checkpoint']
            if key not in cache:
                evaltag=tag+'_step'+str(step);ep=OUT/'results'/(evaltag+'_evaluation.json')
                if ep.exists():ev=json.loads(ep.read_text())
                else:
                    oldeval=legacy_evaluation(study,unit,name,family,seed,step) if not choice['early_added'] else None
                    n=cls.load(ROOT/key);ev=evaluate(n,mt,data,study,unit,name,seed,evaltag,oldeval);ep.write_text(json.dumps(ev,indent=2));del n;gc.collect();jax.clear_caches()
                cache[key]=ev
            evaluations[rule]=dict(step=step,**cache[key])
        dest.write_text(json.dumps(dict(**meta,evaluations=evaluations,seconds=time.time()-start),indent=2))
        print('REPAIRED',tag,{k:v['step'] for k,v in meta['choices'].items()},'sec',round(time.time()-start,1),flush=True)
        del mt;gc.collect();jax.clear_caches()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--unit',required=True,type=int);p.add_argument('--family',default='original');p.add_argument('--seed',default=0,type=int);p.add_argument('--models',nargs='+');a=p.parse_args();work(a.study,a.unit,a.family,a.seed,a.models)
