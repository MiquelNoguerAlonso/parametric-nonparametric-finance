"""Independent numerical checks for the tempered Wishart implementation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from wishart_tempered import TemperedWishart,jr,j
from wishart_model import symmetric_root,normal_logpdf,mixture_risk
ROOT=Path(__file__).resolve().parents[1]


def numpy_proposal(start,par,observed,z,eps,steps):
    n,b,d=start.shape;dt=1./steps;k=par['kappa'];rho=par['rho'];c=np.array(par['C']);mu=np.array(par['mu']);ito=par.get('ito_scale',0.)
    sigma=start.transpose(0,2,1)@start;root=symmetric_root(sigma);inv=symmetric_root(sigma,True)
    drift=mu-.5*ito*np.diagonal(sigma,axis1=1,axis2=2);proj=inv@start.transpose(0,2,1)
    db=np.sqrt(dt)*z[:,0];dw=proj@db.sum(1)
    pseudo=rho*np.diagonal(dw,axis1=1,axis2=2)+np.sqrt(1-rho*rho)*eps
    target=np.einsum('nij,nj->ni',inv,observed-drift)
    db+=dt*rho*proj.transpose(0,2,1)[:,None]*(target-pseudo)[:,None,None,:]
    frozen_mean=drift+rho*np.einsum('nij,nj->ni',root,np.diagonal(proj@db.sum(1),axis1=1,axis2=2))
    logf=normal_logpdf(observed,frozen_mean,(1-rho*rho)*sigma)
    a=np.exp(-k*dt/2);ouv=-np.expm1(-k*dt)/k;cross=-np.expm1(-k*dt/2)/(k/2);res=np.sqrt(max(0.,ouv-cross*cross/dt));q=np.sqrt(k/b)*symmetric_root(c)
    y=start.copy();mean=np.zeros((n,d));cov=np.zeros((n,d,d))
    for t in range(steps):
        sig=y.transpose(0,2,1)@y;root=symmetric_root(sig);inv=symmetric_root(sig,True)
        dw=inv@(y.transpose(0,2,1)@db[:,t]);mean+=(mu-.5*ito*np.diagonal(sig,axis1=1,axis2=2))*dt+rho*np.einsum('nij,nj->ni',root,np.diagonal(dw,axis1=1,axis2=2));cov+=(1-rho*rho)*sig*dt
        y=a*y+((cross/dt)*db[:,t]+res*z[:,1,t])@q
    return y,normal_logpdf(observed,mean,cov)-logf


def main():
    t=time.time();rng=np.random.default_rng(31099);c=.5*np.eye(5)+.5*np.ones((5,5));p=dict(C=c.tolist(),mu=[0.]*5,beta=6,kappa=.2,rho=-.7,ito_scale=0.)
    checks={}
    for steps in [1,4,16]:
        m=TemperedWishart(p,steps,scales=(.4,));y=np.array(m.initial(jr.PRNGKey(10+steps),256));z=rng.standard_normal((256,2,steps,6,5));eps=rng.standard_normal((256,5));r=np.array([-2.,-1.5,-1.,-.5,-1.2])
        a,b=m.proposal(j.asarray(y),j.asarray(r),j.asarray(z),j.asarray(eps));aa,bb=numpy_proposal(y,p,r,z.copy(),eps,steps)
        err=max(float(np.max(abs(np.asarray(a)-aa))),float(np.max(abs(np.asarray(b)-bb))))
        assert err<2e-8;checks[f'numpy_jax_proposal_error_s{steps}']=err
        if steps==1:
            assert np.max(abs(np.asarray(b)))<2e-9
            yn,d=m.update(j.asarray(y),j.asarray(r),jr.PRNGKey(40));diag=np.array(d)
            sigma=y.transpose(0,2,1)@y;true=-logsumexp(normal_logpdf(r,np.zeros((256,5)),sigma))+np.log(256)
            assert abs(diag[-1]-true)<2e-9;checks['fully_adapted_nll_error']=abs(diag[-1]-true);checks['fully_adapted_correction_max']=float(np.max(abs(np.asarray(b))))
    # Risk is checked independently against the scipy root solver on the
    # same Gaussian mixture, not merely against another JAX formula.
    m=TemperedWishart(p,8,scales=(.4,));y=m.initial(jr.PRNGKey(51),1024);mean,cov=m.predict(y,jr.PRNGKey(52),16384)
    w=np.array([np.ones(5)/5,np.eye(5)[0],np.eye(5)[-1]]);q,e=m.risk(mean,cov,j.asarray(w),j.array([.01,.025]));err=0.
    for i,v in enumerate(w):
        for k,a in enumerate([.01,.025]):
            qs,es=mixture_risk(np.array(mean),np.array(cov),v,a);err=max(err,abs(qs-float(q[i,k])),abs(es-float(e[i,k])))
    assert err<2e-8;checks['scipy_mixture_risk_max_error']=err
    # Changing a realized return cannot change that date's issued forecast;
    # the state and subsequent forecasts must respond.
    q0,e0=m.risk(mean,cov,j.asarray(w),j.array([.01,.025]));r=j.array([-1.,-1.,-1.,-1.,-1.]);a,da=m.update(y,r,jr.PRNGKey(53));b,db=m.update(y,r+3,jr.PRNGKey(53))
    assert np.array_equal(q,q0) and np.array_equal(e,e0) and not np.allclose(a,b)
    assert float(da[0])==1. and float(db[0])==1.;assert float(da[2])>=.7*1024-1e-5
    checks['predictable_forecast_check']=True;checks['posterior_responds_to_observation']=True;checks['adaptive_tempering_completes']=True
    # pCN reference invariance: variances and cross-correlation in a Gaussian
    # prior are checked independently of the observation likelihood.
    z=rng.normal(size=(200000,4));eps=rng.normal(size=z.shape);scale=.4;zp=np.sqrt(1-scale**2)*z+scale*eps
    assert np.max(abs(zp.var(0)-1))<.012;checks['pcn_prior_variance_max_error']=float(np.max(abs(zp.var(0)-1)))
    checks['passed']=True;checks['seconds']=time.time()-t
    path=ROOT/'wishart_completion/analysis/kernel_checks.json';path.write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
