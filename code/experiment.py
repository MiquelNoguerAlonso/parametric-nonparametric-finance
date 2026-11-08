"""Run every reported simulation or chronological CRSP experiment.

Examples:
 python code/experiment.py --study t --seeds 0 1
 python code/experiment.py --study metat --seeds 0 1
 python code/experiment.py --study heston --seeds 0 1
 python code/experiment.py --study crsp --seeds 0 1
Run from the package root. Existing completed model records are reused.
"""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import jax
from models import Neural,MetaT,tmap

ROOT=Path(__file__).resolve().parents[1]
for folder in ['results','checkpoints','data']:(ROOT/folder).mkdir(exist_ok=True)

def es(loss,a=.01):
    v=np.sort(np.asarray(loss));m=len(v)*a;k=int(np.floor(m));fraction=m-k
    return float((v[-k:].sum() if k else 0)+(fraction*v[-k-1] if fraction else 0))/m

def hill(x,q=.01):
    v=np.sort(np.abs(x));k=max(10,int(len(v)*q));return float(1/np.log(v[-k:]/max(v[-k-1],1e-12)).mean())

def sample_law(law,n,seed):
    r=np.random.default_rng(seed);P=np.array([[1.,.5],[.5,1.]]);nu=3 if law=='t' else 4
    y=(r.standard_normal((n,2))@np.linalg.cholesky(P).T)*np.sqrt(nu/r.chisquare(nu,(n,1)))
    return y if law=='t' else tmap(y,4,np.array([3.,8.]))

def radial(x,X,q=.05):
    rr=np.linalg.norm(X,axis=1);u=np.quantile(rr,1-q)
    xi,_,beta=st.genpareto.fit(rr[rr>u]-u,floc=0.)
    info={'q':q,'xi':float(xi),'beta':float(beta),'u':float(u),'finite_es':bool(xi<1)}
    if xi>=1:return None,info
    r=np.linalg.norm(x,axis=1);order=np.argsort(r);rank=np.empty(len(r));rank[order]=(np.arange(len(r))+.5)/len(r)
    rn=np.quantile(rr,np.minimum(rank,1-q));tail=rank>1-q
    rn[tail]=u+st.genpareto.ppf((rank[tail]-1+q)/q,xi,scale=beta)
    return x*(rn/np.maximum(r,1e-12))[:,None],info

def meta_samples(X,V,T,seed,path,residual=False):
    mp=path.with_suffix('.meta.json')
    if mp.exists():m=MetaT.load(mp)
    else:m=MetaT().fit(X);m.save(mp)
    info={'marginal_df':m.marg[:,0].tolist(),'copula_df':m.nu,'P':m.P.tolist()}
    if not residual:return m.sample(seed=8000+seed),m.logpdf(T),info
    if path.exists():n=Neural.load(path)
    else:n=Neural('flow',X.shape[1],seed,base='t-elliptical',fixed_nu=m.nu,P=m.P).fit(m.to_y(X),m.to_y(V),path)
    y=n.sample(seed=8000+seed);yt=m.to_y(T)
    return m.from_y(y),n.logpdf(yt)+m.logjac(T,yt),dict(info,validation_step=n.best_step)

def fit_sample(name,X,V,T,seed,path):
    r=np.random.default_rng(8000+seed)
    if name=='Historical':return X[r.integers(len(X),size=100000)],None,{}
    if name=='Gaussian':
        mean=X.mean(0);cov=np.cov(X,rowvar=False)+np.eye(X.shape[1])*1e-7
        return r.multivariate_normal(mean,cov,100000),st.multivariate_normal.logpdf(T,mean,cov),{}
    if name in ['Meta-t','Flow-meta-t']:return meta_samples(X,V,T,seed,path,name=='Flow-meta-t')
    mapping={'Flow-G':('flow','gaussian'), 'Flow-t':('flow','t-elliptical'), 'Flow-L':('flow','laplace'),
        'VAE-G':('vae','gaussian'),'VAE-t':('vae','t-elliptical'),'VAE-H':('vae-scale','gaussian'),
        'WGAN-G':('gan','gaussian'),'WGAN-t':('gan','t-elliptical'),'WGAN-i':('gan','t-independent')}
    kind,base=mapping[name]
    if path.exists():n=Neural.load(path)
    else:n=Neural(kind,X.shape[1],seed,base=base).fit(X,V,path)
    info={'validation_step':n.best_step,'fit_seconds':n.fit_seconds}
    if name=='Flow-t':info['base_df']=float(n.nu(n.p))
    if hasattr(n,'lipschitz_bound'):info['standardized_lipschitz_bound']=n.lipschitz_bound
    out=n.sample(seed=8000+seed);lp=n.logpdf(T)
    return out,lp,info

def heston_bank(seed=993,n=20000,sub=4,years=5):
    p=ROOT/'data'/f'heston_bank_{seed}_{n}_{sub}_{years}.npz'
    if p.exists():return np.load(p)['V']
    r=np.random.default_rng(seed);a=.96;b=.04/a;v=r.gamma(a,b,(n,5));dt=1/(252*sub)
    C=np.linalg.cholesky(.4*np.eye(5)+.6*np.ones((5,5)));diagnostics=[]
    for i in range(years*252*sub):
        vp=np.maximum(v,0);v+=3*(.04-vp)*dt+.5*np.sqrt(vp*dt)*(r.standard_normal(v.shape)@C.T)
        if (i+1)%(252*sub)==0:diagnostics.append({'year':(i+1)//(252*sub),'mean':float(v.mean()),'var':float(v.var()),'cross_corr':float(np.corrcoef(v.T)[0,1])})
    np.savez_compressed(p,V=v);p.with_suffix('.json').write_text(json.dumps(diagnostics,indent=2))
    return v

def heston(n,seed,sub=20,bank=None):
    bank=heston_bank() if bank is None else bank;r=np.random.default_rng(seed)
    V=bank[r.integers(len(bank),size=n)].copy();out=np.zeros((n,5,5));dt=1/(252*sub)
    cv=np.linalg.cholesky(.4*np.eye(5)+.6*np.ones((5,5)))
    cp=np.linalg.cholesky(.5*np.eye(5)+.5*np.ones((5,5)))
    for day in range(5):
        for _ in range(sub):
            db=r.standard_normal((n,5))@cv.T;dp=r.standard_normal((n,5))@cp.T
            dw=-.7*db+np.sqrt(1-.7**2)*dp;vp=np.maximum(V,0.)
            out[:,:,day]+=-.5*vp*dt+np.sqrt(vp*dt)*dw
            V+=3*(.04-vp)*dt+.5*np.sqrt(vp*dt)*db
    return (out.reshape(n,25)/np.sqrt(.04/252)).astype(np.float32)

def stylised(x):
    a=np.abs(x);days=[(b*5+j,b*5+j+1) for b in range(5) for j in range(4)]
    pairs=[(p*5+j,q*5+j) for p in range(5) for q in range(p+1,5) for j in range(5)]
    cc=lambda y,z:float(np.corrcoef(y,z)[0,1])
    return {'kurtosis':float(st.kurtosis(x.ravel())),
        'cluster':np.mean([cc(a[:,i],a[:,j]) for i,j in days]),
        'leverage':np.mean([cc(x[:,i],a[:,j]) for i,j in days]),
        'cross_vol':np.mean([cc(a[:,i],a[:,j]) for i,j in pairs])}

def synthetic(study,seeds,datasets):
    law=heston if study=='heston' else lambda n,seed:sample_law(study,n,seed)
    d=25 if study=='heston' else 2
    weights={'equal':np.ones(d)/5 if study=='heston' else np.ones(2)/2}
    if study!='heston':weights.update(first=np.array([1.,0.]),second=np.array([0.,1.]))
    targetpath=ROOT/'data'/f'{study}_truth.npz'
    if targetpath.exists():truth=np.load(targetpath)['X']
    else:truth=law(400000,777);np.savez_compressed(targetpath,X=truth)
    true={};blocks=np.array_split(truth,20)
    for pn,w in weights.items():
        for alpha in [.01,.001]:
            value=es(-truth@w,alpha)
            if study=='t':
                s=np.sqrt(w@np.array([[1.,.5],[.5,1.]])@w);q=st.t.ppf(alpha,3)
                value=float(s*st.t.pdf(q,3)/alpha*(3+q*q)/2)
            true[f'{pn}_{alpha}']={'es':value,'mc_block_se':float(np.std([es(-b@w,alpha) for b in blocks],ddof=1)/np.sqrt(20))}
    if study=='heston':true['stylised']=stylised(truth[:100000])
    (ROOT/'results'/f'{study}_truth.json').write_text(json.dumps(true,indent=2))
    T=law(20000,999)
    names=['Historical','Gaussian','Meta-t','Flow-G','Flow-t','VAE-G','VAE-t','WGAN-G','WGAN-t','WGAN-i']
    if study=='metat':names.append('Flow-meta-t')
    if study=='heston':names=['Historical','Gaussian','Flow-G','Flow-t','Flow-L','VAE-G','VAE-H','WGAN-G','WGAN-t']
    for ds in range(datasets):
        X=law(4000,1000+ds);V=law(1000,2000+ds)
        for seed in seeds:
            fitseed=10000*ds+seed
            for name in names:
                tag=f'{study}_{ds}_{seed}_{name}';result=ROOT/'results'/f'{tag}.json'
                if result.exists():continue
                start=time.time();samples,lp,info=fit_sample(name,X,V,T,fitseed,ROOT/'checkpoints'/f'{tag}.npz')
                if not np.isfinite(samples).all():raise ValueError(f'Nonfinite samples {tag}')
                record={'study':study,'dataset':ds,'seed':seed,'model':name,'nll':None if lp is None else float(-np.mean(lp)),
                    'nll_kind':'IWAE64 Monte Carlo estimate' if name.startswith('VAE') else 'exact density' if lp is not None else 'none',
                    'hill1':hill(samples[:,0]),'hill01':hill(samples[:,0],.001),'info':info,'metrics':[]}
                if d==2:record['hill2']=hill(samples[:,1])
                for pn,w in weights.items():
                    loss=-samples@w;ref=-truth@w;w1=float(st.wasserstein_distance(loss,ref))
                    for alpha in [.01,.001]:
                        e=es(loss,alpha);et=true[f'{pn}_{alpha}']['es'];ep=es(-X@w,alpha)
                        record['metrics'].append({'portfolio':pn,'alpha':alpha,'es':e,'truth':et,'sample_es':ep,
                            'total_error':e-et,'fit_error':e-ep,'sample_error':ep-et,'w1_mc':w1,'ratio_mc':alpha*abs(e-et)/w1})
                if study!='heston':
                    record['transplant']=[]
                    for q in [.025,.05,.10]:
                        repaired,fit=radial(samples,X,q);fit['equal_es']=None if repaired is None else es(-repaired@weights['equal'])
                        fit['total_error']=None if repaired is None else fit['equal_es']-true['equal_0.01']['es']
                        record['transplant'].append(fit)
                    if ds==0 and seed==0:np.savez_compressed(ROOT/'data'/f'curve_{study}_{name}.npz',loss=np.sort(-samples@weights['equal']))
                else:record['stylised']=stylised(samples)
                record['seconds']=time.time()-start;result.write_text(json.dumps(record,indent=2))
                print(tag,'NLL',record['nll'],'ES error',record['metrics'][0]['total_error'],'sec',round(record['seconds'],1),flush=True)
                del samples,lp;gc.collect();jax.clear_caches()

def ewma(X,lam=.94):
    v=np.var(X[:252],axis=0);out=np.empty_like(X)
    for t,x in enumerate(X):out[t]=np.sqrt(np.maximum(v,1e-10));v=lam*v+(1-lam)*x*x
    return out

def forecast(samples,scales,returns,dates,model,seed,fold,clip_support=False):
    # Scenarios are independent of realized test outcomes. Common scenario indices
    # across dates reduce simulation noise in paired chronological comparisons.
    samples=samples[:20000];portfolios={'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}
    out=[];alphas=[.01,.025]
    for pn,w in portfolios.items():
        for start in range(0,len(scales),32):
            scale=scales[start:start+32]
            if clip_support:
                loss=-np.maximum(samples[:,None,:]*scale[None,:,:],-100.)@w
            else:
                loss=-samples@(scale*w).T
            actual=-returns[start:start+32]@w
            for alpha in alphas:
                k=int(len(samples)*alpha);cut=len(samples)-k
                part=np.partition(loss,[cut-1,cut],axis=0)
                var=part[cut-1];short=part[cut:].mean(0)
                assert np.all(short>0)
                score=np.log(short)+var/short-1+np.maximum(actual-var,0)/(alpha*short)
                residual=short-var-np.maximum(actual-var,0)/alpha
                for j in range(len(scale)):
                    out.append({'date':str(pd.Timestamp(dates[start+j]).date()),'fold':fold,'seed':seed,'model':model,'portfolio':pn,'alpha':alpha,
                        'loss':float(actual[j]),'var':float(var[j]),'es':float(short[j]),'hit':int(actual[j]>var[j]),'fz0':float(score[j]),'es_residual':float(residual[j])})
    return pd.DataFrame(out)

def crsp(seeds,years):
    f=pd.read_csv(ROOT/'data'/'crsp_size_portfolios.csv',index_col=0,parse_dates=True)
    X=f.values*100.;scales=ewma(X);Z=X/scales
    names=['Historical','Gaussian','Meta-t','Flow-G','Flow-t','Flow-meta-t','VAE-G','VAE-H','WGAN-G','WGAN-t']
    for year in years:
        train=(f.index.year>=year-11)&(f.index.year<=year-2)&(f.index>=pd.Timestamp('2006-01-01'))
        val=f.index.year==year-1;test=f.index.year==year
        assert f.index[train].max()<f.index[val].min()<f.index[test].min()
        for seed in seeds:
            for name in names:
                tag=f'crsp_{year}_{seed}_{name}';result=ROOT/'results'/f'{tag}.json';csv=ROOT/'results'/f'{tag}.csv'
                if result.exists() and csv.exists():continue
                start=time.time();samples,lp,info=fit_sample(name,Z[train],Z[val],Z[test],10000+year*10+seed,ROOT/'checkpoints'/f'{tag}.npz')
                preds=forecast(samples,scales[test],X[test],f.index[test],name,seed,year);preds.to_csv(csv,index=False)
                record={'study':'crsp','model':name,'fold':year,'seed':seed,'train_start':str(f.index[train].min().date()),
                    'train_end':str(f.index[train].max().date()),'validation_start':str(f.index[val].min().date()),
                    'validation_end':str(f.index[val].max().date()),'test_start':str(f.index[test].min().date()),'test_end':str(f.index[test].max().date()),
                    'n_train':int(train.sum()),'n_val':int(val.sum()),'n_test':int(test.sum()),'info':info,
                    'nll_percent_return':None if lp is None else float(np.mean(-lp+np.log(scales[test]).sum(1))),
                    'seconds':time.time()-start}
                result.write_text(json.dumps(record,indent=2))
                print(tag,'NLL',record['nll_percent_return'],'sec',round(record['seconds'],1),flush=True)
                del samples,lp,preds;gc.collect();jax.clear_caches()

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['t','metat','heston','crsp'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1]);p.add_argument('--datasets',type=int,default=3)
    p.add_argument('--years',nargs='+',type=int,default=list(range(2016,2026)));a=p.parse_args()
    if a.study=='crsp':crsp(a.seeds,a.years)
    else:synthetic(a.study,a.seeds,a.datasets)

if __name__=='__main__':main()
