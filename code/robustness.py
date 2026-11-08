"""Simulator refinement, scenario Monte Carlo, and support diagnostics."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import jax
from models import Neural,MetaT
from experiment import heston_bank,heston,es,stylised,ewma,forecast
ROOT=Path(__file__).resolve().parents[1]

def paired_heston(n=100000,seed=312):
    r=np.random.default_rng(seed);bank=heston_bank();v=bank[r.integers(len(bank),size=n)].copy();v2=v.copy()
    out=np.zeros((n,5,5));out2=out.copy();dt=1/(252*40)
    cv=np.linalg.cholesky(.4*np.eye(5)+.6*np.ones((5,5)));cp=np.linalg.cholesky(.5*np.eye(5)+.5*np.ones((5,5)))
    for day in range(5):
        for step in range(20):
            bs=[];ws=[]
            for small in range(2):
                b=r.standard_normal((n,5))@cv.T;p=r.standard_normal((n,5))@cp.T
                w=-.7*b+np.sqrt(1-.7**2)*p;bs.append(b);ws.append(w);vp=np.maximum(v2,0)
                out2[:,:,day]+=-.5*vp*dt+np.sqrt(vp*dt)*w
                v2+=3*(.04-vp)*dt+.5*np.sqrt(vp*dt)*b
            b=(bs[0]+bs[1])/np.sqrt(2);w=(ws[0]+ws[1])/np.sqrt(2);vp=np.maximum(v,0)
            out[:,:,day]+=-.5*vp*2*dt+np.sqrt(vp*2*dt)*w
            v+=3*(.04-vp)*2*dt+.5*np.sqrt(vp*2*dt)*b
    return out.reshape(n,25)/np.sqrt(.04/252),out2.reshape(n,25)/np.sqrt(.04/252)

def simulator():
    x,y=paired_heston();a=-x.sum(1)/5;b=-y.sum(1)/5
    record={'coupled_paths':len(x),'coarse_substeps':20,'fine_substeps':40,'risk':[]}
    for alpha in [.01,.001]:
        differences=[es(bb,alpha)-es(aa,alpha) for aa,bb in zip(np.array_split(a,20),np.array_split(b,20))]
        se=float(np.std(differences,ddof=1)/np.sqrt(20));diff=es(b,alpha)-es(a,alpha)
        record['risk'].append({'alpha':alpha,'coarse_es':es(a,alpha),'fine_es':es(b,alpha),'difference':diff,'paired_block_se':se})
    del x,y;gc.collect()
    refined=heston_bank(seed=994,n=20000,sub=8,years=5)
    x=heston(100000,995,bank=refined);record['refined_bank']={'mean_variance':float(refined.mean()),'variance_of_variance':float(refined.var()),
       'cross_correlation':float(np.corrcoef(refined.T)[0,1]),'weekly_es_1':es(-x.sum(1)/5),'stylised':stylised(x)}
    (ROOT/'results'/'heston_refinement.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

def scenarios():
    f=pd.read_csv(ROOT/'data'/'crsp_size_portfolios.csv',index_col=0,parse_dates=True);X=f.values*100;sc=ewma(X);Z=X/sc
    records=[];supports=[]
    names=['Historical','Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']
    for year in [2020,2025]:
        train=(f.index.year>=year-11)&(f.index.year<=year-2)&(f.index.year>=2006);test=f.index.year==year
        for name in names:
            tag=f'crsp_{year}_0_{name}';path=ROOT/'checkpoints'/f'{tag}.npz'
            if name in ['Meta-t','Flow-meta-t']:mt=MetaT.load(path.with_suffix('.meta.json'))
            if name not in ['Historical','Gaussian','Meta-t']:neural=Neural.load(path)
            for seed in [11,22,33]:
                r=np.random.default_rng(seed)
                if name=='Historical':s=Z[train][r.integers(train.sum(),size=100000)]
                elif name=='Gaussian':s=r.multivariate_normal(Z[train].mean(0),np.cov(Z[train],rowvar=False)+1e-7*np.eye(5),100000)
                elif name=='Meta-t':s=mt.sample(seed=seed)
                elif name=='Flow-meta-t':s=mt.from_y(neural.sample(seed=seed))
                else:s=neural.sample(seed=seed)
                worst=s*np.max(sc[test],axis=0);count=int(np.any(worst < -100,axis=1).sum())
                supports.append({'year':year,'model':name,'scenario_seed':seed,'n':len(s),'outside_simple_return_support':count,
                   'worst_coordinate_scale_upper_envelope':True})
                pred=forecast(s,sc[test],X[test],f.index[test],name,0,year);a=pred[(pred.portfolio=='equal')&(pred.alpha==.01)]
                records.append({'year':year,'model':name,'scenario_seed':seed,'mean_fz0':float(a.fz0.mean()),'mean_es':float(a.es.mean()),'coverage':float(a.hit.mean())})
            print('scenario sensitivity',year,name,flush=True);gc.collect();jax.clear_caches()
    pd.DataFrame(records).to_csv(ROOT/'results'/'crsp_scenario_sensitivity.csv',index=False)
    pd.DataFrame(supports).to_csv(ROOT/'results'/'crsp_support_diagnostics.csv',index=False)

def support_clip():
    f=pd.read_csv(ROOT/'data'/'crsp_size_portfolios.csv',index_col=0,parse_dates=True);X=f.values*100;sc=ewma(X)
    s=pd.read_csv(ROOT/'results'/'crsp_support_diagnostics.csv')
    a=pd.read_csv(ROOT/'results'/'crsp_scenario_sensitivity.csv');rows=[]
    for _,r in s.iterrows():
        year=int(r.year);seed=int(r.scenario_seed);name=r.model
        base=a[(a.year==year)&(a.model==name)&(a.scenario_seed==seed)].iloc[0]
        row=dict(year=year,model=name,scenario_seed=seed,base_fz0=float(base.mean_fz0),clipped_fz0=float(base.mean_fz0),base_es=float(base.mean_es),clipped_es=float(base.mean_es))
        if r.outside_simple_return_support>0:
            path=ROOT/'checkpoints'/f'crsp_{year}_0_{name}.npz';neural=Neural.load(path);samples=neural.sample(seed=seed)
            if name=='Flow-meta-t':samples=MetaT.load(path.with_suffix('.meta.json')).from_y(samples)
            test=f.index.year==year;pred=forecast(samples,sc[test],X[test],f.index[test],name,0,year,clip_support=True)
            pred=pred[(pred.portfolio=='equal')&(pred.alpha==.01)]
            row.update(clipped_fz0=float(pred.fz0.mean()),clipped_es=float(pred.es.mean()))
            jax.clear_caches();gc.collect()
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results'/'crsp_support_clipping.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['heston','crsp','support'],required=True);a=p.parse_args()
    if a.part=='heston':simulator()
    elif a.part=='crsp':scenarios()
    else:support_clip()
