"""Pointwise density and joint-event records for unchanged statistical laws."""
import argparse,json,gc
import numpy as np
from scipy import stats
from models import MetaT
from round4_data import get_data,logdet,joint_event,ROOT,OUT

def work(study,unit,family='original'):
    data=get_data(study,unit,family)
    for name in ['Historical','Gaussian','Meta-t']:
        tag=f'{family}_{study}_{unit}_{name}';dest=OUT/'results'/(tag+'.json')
        if dest.exists():continue
        mt=None
        if name=='Meta-t':
            base=ROOT/'referee_round3/checkpoints'/f'coord_{study}_{unit}.meta.json' if family=='coord' else ROOT/('referee_round2' if study=='heston' else 'expansion')/'checkpoints'/f'{study}_{unit}_0_Meta-t.meta.json'
            mt=MetaT.load(base);lp=mt.logpdf(data['T'])
        elif name=='Gaussian':mu=data['X'].mean(0);cov=np.cov(data['X'],rowvar=False)+1e-7*np.eye(len(mu));lp=stats.multivariate_normal.logpdf(data['T'],mu,cov)
        else:lp=None
        r=dict(study=study,unit=unit,family=family,model=name)
        if lp is not None:
            nll=-lp+logdet(data);p=OUT/'pointwise'/(tag+'.npz');np.savez_compressed(p,nll=nll);r.update(nll=float(nll.mean()),pointwise=str(p.relative_to(ROOT)))
        else:r['nll']=None
        if study.startswith('crsp') and study!='crsp5cov':
            if name=='Historical':z=data['X']
            elif name=='Meta-t':z=mt.sample(100000,190000+10*unit)
            else:z=np.random.default_rng(190000+10*unit).multivariate_normal(mu,cov,100000)
            r['joint_event']=joint_event(z,data)
        if study in ['t','metat'] and name=='Meta-t':
            x=data['T']
            if study=='t':oracle=-stats.multivariate_t.logpdf(x,shape=[[1,.5],[.5,1]],df=3)
            else:
                o=MetaT();o.marg=np.array([[3,0,1],[8,0,1]]);o.nu=4.;o.P=np.array([[1.,.5],[.5,1.]]);o.d=2;oracle=-o.logpdf(x)
            np.savez_compressed(OUT/'pointwise'/f'oracle_{study}.npz',nll=oracle)
        dest.write_text(json.dumps(r,indent=2));print('STATIC',tag,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study');p.add_argument('--unit',type=int);p.add_argument('--family',default='original');a=p.parse_args()
    if a.study:work(a.study,a.unit,a.family)
    else:
        for s in ['t','metat','heston','crsp5','crsp100','crsp5cov']:
            for u in (range(2016,2026) if s.startswith('crsp') else range(10)):work(s,u)
        for s in ['crsp5','crsp100']:
            for u in range(2016,2026):work(s,u,'coord')
