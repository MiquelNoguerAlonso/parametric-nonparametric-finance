"""Independent accounting, predictability, likelihood and inference checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from dynamic_benchmarks import unpack,variance,objective
from revision_analysis import mcs
from models import MetaT

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/revision';checks=[]
def check(name,condition,detail):
    checks.append(dict(check=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError((name,detail))

def main():
    d=pd.read_csv(OUT/'dynamic_forecasts.csv.gz');p=pd.read_csv(OUT/'dynamic_parameters.csv')
    check('New forecast panel balanced and unique',len(d)==2514*3*3*2 and not d.duplicated(['date','model','portfolio','alpha']).any(),len(d))
    check('All retained GARCH optimizers successful',p.success.all() and len(p)==30,len(p))
    check('GARCH parameter restrictions',((p.arch+p.beta<.9999)&(p.arch>0)&(p.beta>0)&(p.omega>0)&p.nu.between(2.05,200)).all(),float((p.arch+p.beta).max()))
    check('Finite EVT means',(p.evt_xi<1).all(),float(p.evt_xi.max()))
    score=np.log(d.es)+d['var']/d.es-1+np.maximum(d.loss-d['var'],0)/(d.alpha*d.es)
    check('New forecast FZ0 accounting',np.max(np.abs(score-d.fz0))<1e-10,float(np.max(np.abs(score-d.fz0))))
    check('New forecast ES ordering',(d.es>=d['var']).all(),float((d.es-d['var']).min()))
    # Compare recursive JAX likelihood against a scalar SciPy calculation.
    x=np.random.default_rng(17).standard_t(5,250);par=np.array([.1,-2.,.5,3.,7.]);initial=1.1
    mu,omega,a,b,nu=map(float,unpack(par));v=[];current=initial
    for xt in x:v.append(current);current=omega+a*(xt-mu)**2+b*current
    v=np.array(v);vj=np.asarray(variance(par,x,initial))
    check('GARCH scalar recursion',np.max(np.abs(v-vj))<1e-11,float(np.max(np.abs(v-vj))))
    lp=st.t.logpdf((x-mu)/np.sqrt(v)*np.sqrt(nu/(nu-2)),nu)-.5*np.log(v)-.5*np.log((nu-2)/nu)
    check('GARCH conditional likelihood equals SciPy',abs(float(objective(par,x,initial))+lp.mean())<1e-10,float(objective(par,x,initial)))
    altered=x.copy();altered[150:]+=10
    check('Variance states predictable',np.allclose(variance(par,x,initial)[:151],variance(par,altered,initial)[:151]),'future shocks cannot change earlier states')
    # Independently integrate Student tail rather than compare to same formula.
    from scipy.integrate import quad
    alpha=.01;nu=7.;q=st.t.isf(alpha,nu)
    integral=quad(lambda z:z*st.t.pdf(z,nu),q,np.inf,epsabs=1e-11)[0]/alpha
    closed=(nu+q*q)/(nu-1)*st.t.pdf(q,nu)/alpha
    check('Analytic GARCH Student ES',abs(integral-closed)<1e-9,abs(integral-closed))
    n=0;maxdiag=0.;mineig=np.inf;boundaries=0
    for f in (ROOT/'checkpoints').glob('*.meta.json'):
        z=json.loads(f.read_text());c=np.array(z['P']);m=np.array(z['marg']);n+=1
        maxdiag=max(maxdiag,float(np.max(np.abs(np.diag(c)-1))));mineig=min(mineig,float(np.linalg.eigvalsh(c).min()))
        boundaries+=int(np.any((m[:,0]<=1.05)|(m[:,0]>=200)))
    check('All stored meta-t matrices are correlations',maxdiag<1e-12 and mineig>0,dict(count=n,max_diagonal_error=maxdiag,min_eigenvalue=mineig))
    check('Clipping inactive in every archived marginal fit',boundaries==0,boundaries)
    fitted=MetaT().fit(np.random.default_rng(41).standard_t(4,(1200,2)))
    check('New constrained marginal fitting',np.all((fitted.marg[:,0]>=1.05)&(fitted.marg[:,0]<=200)) and np.all(fitted.marg[:,2]>0),fitted.marg.tolist())
    a=pd.read_csv(OUT/'synthetic_decomposition.csv')
    err=abs(a.fit_mse+a.sample_mse+a['cross']-a.total_mse).max()
    check('Reported MSE decomposition includes cross term',err<1e-10,float(err))
    check('Dataset is simulation replication unit',(a.n_datasets==3).all(),a.n_datasets.unique().tolist())
    inf=pd.read_csv(OUT/'inference.csv');reject=(inf.low>0)|(inf.high<0)
    # Bootstrap interpolation and plus-one corrections can disagree only at a boundary.
    check('Intervals invert tests away from boundary',np.all(reject[abs(inf.p-.05)>.001]==(inf.p[abs(inf.p-.05)>.001]<.05)),int(reject.sum()))
    oracle=pd.read_csv(OUT/'oracle_nll.csv').set_index('study')
    check('Student entropy and common-sample oracle',abs(oracle.loc['t','population_entropy']-3.36070269685)<1e-10 and oracle.loc['t','oracle_nll']<3.3542006051,oracle.loc['t'].to_dict())
    # TR eliminates a manifestly inferior forecast and is invariant to common
    # date-specific score shifts and permutations, as relative scores require.
    rng=np.random.default_rng(52);base=rng.normal(size=400)
    scores=pd.DataFrame({'a':base+rng.normal(0,.1,400),'b':base+rng.normal(0,.1,400),'bad':base+4+rng.normal(0,.1,400)})
    one=mcs(scores,B=499);two=mcs(scores.add(rng.normal(size=400),axis=0)[['bad','b','a']],B=499)
    members=lambda g:set(g.loc[g.p_mcs>=.05,'model'])
    check('MCS direction and relative-score invariance',members(one)==members(two)=={'a','b'},sorted(members(one)))
    (OUT/'revision_checks.json').write_text(json.dumps(checks,indent=2))
    print(f'{len(checks)} revision checks passed')

if __name__=='__main__':main()
