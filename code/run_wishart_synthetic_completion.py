"""Independent synthetic grid audit; never changes fitted parameters."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.optimize import brentq
from wishart_coupled import CoupledWishart,jr
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wishart_completion'

def risk(loc,sd,a):
    q=brentq(lambda q:ndtr((loc-q)/sd).mean()-a,np.min(loc-12*sd),np.max(loc+12*sd),xtol=1e-11)
    zz=(q-loc)/sd;tail=ndtr(-zz);ep=loc*tail+sd*np.exp(-.5*zz**2)/np.sqrt(2*np.pi);es=ep.mean()/a
    influence=(ep-q*tail)/a+q-es
    return dict(var=q,es=es,mc_se=float(influence.std(ddof=1)/np.sqrt(len(loc)))),influence

def run(unit,bank,n,finest,batch=1024):
    tag=f'synthetic_d{unit}_b{bank}_n{n}_s{finest}';path=OUT/'results'/(tag+'.json')
    if path.exists():print('EXISTS',tag,flush=True);return
    start=time.time();pf=ROOT/'wishart_extension/fits'/f'heston_{unit}.json';par=json.loads(pf.read_text())['parameters']
    grids=[s for s in [4,8,16,32,64,128] if s<=finest];m=CoupledWishart(par,finest,grids);key=jr.PRNGKey(1000000+unit*100+bank)
    parts={s:[] for s in grids};end_error=0.
    cp=OUT/'checkpoints'/tag;cp.mkdir(exist_ok=True)
    for k in range((n+batch-1)//batch):
        f=cp/f'{k}.npz'
        if f.exists():data=np.load(f); vals={s:data[f'p{s}'] for s in grids};end_error=max(end_error,float(data['end_error']))
        else:
            raw=m.coupled(jr.fold_in(key,k),min(batch,n-k*batch));vals={};end=np.asarray(raw[-1][0]);er=0.
            for s,(y,mean,cov) in zip(grids,raw):
                er=max(er,float(np.max(abs(np.asarray(y)-end))));loc=-np.asarray(mean).mean(1);sd=np.sqrt(np.asarray(cov).mean((1,2)));vals[s]=np.stack([loc,sd],axis=1)
            end_error=max(end_error,er);np.savez_compressed(f,**{f'p{s}':v for s,v in vals.items()},end_error=er)
        for s in grids:parts[s].append(vals[s])
        if (k+1)%8==0:print(tag,k+1,'batches',round(time.time()-start,1),flush=True)
    reference=json.loads((ROOT/'expansion/data/heston_reference.json').read_text());rows=[];pairs=[];influences={};estimates={}
    for s in grids:
        arr=np.concatenate(parts[s]);np.savez_compressed(OUT/'results'/f'{tag}_grid{s}.npz',loc=arr[:,0],sd=arr[:,1])
        for a in [.01,.001]:
            r,inf=risk(arr[:,0],arr[:,1],a);estimates[s,a]=r;influences[s,a]=inf
            rows.append(dict(substeps=s,alpha=a,**r,truth=reference[f'equal_{a}']['es'],error=r['es']-reference[f'equal_{a}']['es']))
    for s,t in zip(grids[:-1],grids[1:]):
        for a in [.01,.001]:
            diff=influences[t,a]-influences[s,a]
            pairs.append(dict(coarse=s,fine=t,alpha=a,es_shift=estimates[t,a]['es']-estimates[s,a]['es'],paired_mc_se=float(diff.std(ddof=1)/np.sqrt(n))))
    out=dict(study='heston',dataset=unit,bank=bank,paths=n,finest=finest,grids=grids,parameters=par,parameter_sha256=hashlib.sha256(pf.read_bytes()).hexdigest(),risk=rows,refinement=pairs,endpoint_max_error=end_error,seconds=time.time()-start,method='nested exact OU factors; conditional Gaussian mixture for five-day portfolio loss')
    assert end_error<1e-9
    path.write_text(json.dumps(out,indent=2));print('DONE',tag,out['refinement'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--units',nargs='+',type=int,required=True);p.add_argument('--bank',type=int,default=0);p.add_argument('--n',type=int,default=32768);p.add_argument('--finest',type=int,default=32);a=p.parse_args()
    for u in a.units:run(u,a.bank,a.n,a.finest)
