"""Training-only fits and retrospective, separately reported Wishart--CIR study."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from wishart_cir import fit,initial,integrals,conditional,risk,project,update
from experiment import heston,es
from expanded_experiment import score
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'wishart_cir_extension'
for sub in ['fits','results','logs','analysis']:(OUT/sub).mkdir(parents=True,exist_ok=True)
def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def training(study,unit):
    if study=='heston':return heston(4000,50000+unit).astype(float).reshape(-1,5,5),None,np.sqrt(.04/252)
    f=pd.read_csv(ROOT/'data/crsp_size_portfolios.csv',index_col=0,parse_dates=True)
    x=f.to_numpy()*100; tr=(f.index.year>=max(2006,unit-11))&(f.index.year<=unit-2)
    return np.lib.stride_tricks.sliding_window_view(x[tr],5,axis=0).copy(),x[tr],0.
def getfit(study,unit):
    path=OUT/'fits'/f'{study}_{unit}.json'
    if path.exists():return json.loads(path.read_text())['parameters']
    start=time.time();x,daily,ito=training(study,unit); p,diag=fit(x,820000+unit,ito,daily)
    save(path,dict(study=study,unit=unit,parameters=p,fit=diag,seconds=time.time()-start,
                   training_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    print('FIT',study,unit,{k:v for k,v in p.items() if k not in ['C','mu']},diag['objective'],round(time.time()-start,1),flush=True)
    return p
def synthetic(unit,n=131072,steps=32):
    path=OUT/'results'/f'heston_{unit}.json'
    if path.exists():return
    start=time.time();p=getfit('heston',unit); levels=[4,8,16,32];assert steps==32
    abl=['full','no_leverage','fixed_component_correlation'];store={(s,a):[[],[]] for s in levels for a in abl}
    rng=np.random.default_rng(910000+unit)
    for k in range(0,n,4096):
        batch=min(4096,n-k);y,v=initial(p,batch,rng)
        sums={(s,a):[np.zeros((batch,5)),np.zeros((batch,5,5))] for s in levels for a in abl}
        for day in range(5):
            y,v,rr=integrals(y,v,p,rng,steps,levels)
            for s in levels:
                for a in abl:
                    m,c=conditional(p,rr[s],a);sums[s,a][0]+=m;sums[s,a][1]+=c
        for key,(m,c) in sums.items():
            loc,sd=project(m,c,np.ones(5)/5);store[key][0].append(loc);store[key][1].append(sd)
    ref=json.loads((ROOT/'expansion/data/heston_reference.json').read_text());x,_,_=training('heston',unit)
    rows=[]; arrays={}
    for (s,a),(loc,sd) in store.items():
        loc=np.concatenate(loc);sd=np.concatenate(sd);arrays[f'{s}_{a}_loc']=loc;arrays[f'{s}_{a}_sd']=sd
        for alpha in [.01,.001]:
            r=risk(loc,sd,alpha);r.update(steps=s,ablation=a,alpha=alpha,truth=ref[f'equal_{alpha}']['es'],empirical=es(-x.sum((1,2))/5,alpha));rows.append(r)
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    save(path,dict(unit=unit,parameters=p,n=n,rows=rows,seconds=time.time()-start))
    print('SYN',unit,[(r['alpha'],round(r['es'],4),round(r['mc_se'],4)) for r in rows if r['steps']==32 and r['ablation']=='full'],round(time.time()-start,1),flush=True)
def filterrun(year,n,steps,bank,predictive=8192,warmup=252,guided=False):
    tag=f'crsp5_{year}_n{n}_s{steps}_b{bank}_p{predictive}_w{warmup}'+('_guided' if guided else '');path=OUT/'results'/(tag+'.json')
    if path.exists():return
    start=time.time();p=getfit('crsp5',year)
    frame=pd.read_csv(ROOT/'data/crsp_size_portfolios.csv',index_col=0,parse_dates=True);x=frame.to_numpy()*100
    te=np.flatnonzero(frame.index.year==year);first=te[0]-warmup
    rng=np.random.default_rng(830000+year*100+bank);y,v=initial(p,n,rng)
    rows=[];diags=[];weights={'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}
    for t in range(first,te[-1]+1):
        if t>=te[0]:
            prng=np.random.default_rng(np.random.SeedSequence([840000,year,bank,t]))
            idx=systematic_uniform(n,predictive,prng)
            _,_,rr=integrals(y[idx],v[idx],p,prng,steps)
            m,c=conditional(p,rr[steps])
            for pn,w in weights.items():
                loc,sd=project(m,c,w)
                for alpha in [.01,.025]:
                    r=risk(loc,sd,alpha);loss=float(-x[t]@w)
                    rows.append(dict(date=str(frame.index[t].date()),portfolio=pn,alpha=alpha,loss=loss,**r,fz0=float(score(loss,r['var'],r['es'],alpha))))
        y,v,diag=update(y,v,p,x[t],rng,steps,guided);diags.append(dict(date=str(frame.index[t].date()),test=bool(t>=te[0]),**diag))
        if (t-first+1)%100==0:
            print(tag,'day',t-first+1,'sec',round(time.time()-start,1),flush=True)
            pd.DataFrame(rows).to_csv(path.with_suffix('.partial.csv'),index=False)
    pd.DataFrame(rows).to_csv(path.with_suffix('.csv.gz'),index=False);pd.DataFrame(diags).to_csv(OUT/'results'/(tag+'_filter.csv.gz'),index=False)
    save(path,dict(year=year,particles=n,steps=steps,bank=bank,predictive=predictive,warmup=warmup,guided=guided,seconds=time.time()-start,parameters=p,n_test=len(te)))
    print('FILTER DONE',tag,'score',np.mean([r['fz0'] for r in rows if r['portfolio']=='equal' and r['alpha']==.01]),flush=True)
def systematic_uniform(n,m,rng):
    return np.floor((np.arange(m)+rng.random())*n/m).astype(int)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit','synthetic','filter']);ap.add_argument('--study',default='heston');ap.add_argument('--units',nargs='+',type=int,required=True)
    ap.add_argument('--particles',type=int,default=2048);ap.add_argument('--steps',type=int,default=8);ap.add_argument('--bank',type=int,default=0);ap.add_argument('--predictive',type=int,default=8192);ap.add_argument('--warmup',type=int,default=252);ap.add_argument('--guided',action='store_true');args=ap.parse_args()
    for unit in args.units:
        if args.mode=='fit':getfit(args.study,unit)
        elif args.mode=='synthetic':synthetic(unit)
        else:filterrun(unit,args.particles,args.steps,args.bank,args.predictive,args.warmup,args.guided)
