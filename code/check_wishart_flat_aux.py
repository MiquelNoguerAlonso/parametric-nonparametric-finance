"""Flattened-ancestor proposal normalizer and antithetic prediction checks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from wishart_tempered import TemperedWishart,jr,j
from wishart_model import normal_logpdf
ROOT=Path(__file__).resolve().parents[1]
t=time.time();c=.5*np.eye(5)+.5*np.ones((5,5));p=dict(C=c.tolist(),mu=[0.]*5,beta=6,kappa=.2,rho=-.7,ito_scale=0.)
base=TemperedWishart(p,1);unique=np.array(base.initial(jr.PRNGKey(300),128));cloud=np.tile(unique,(16,1,1));r=np.array([-2.,-1.5,-1.,-.5,-1.2]);sigma=unique.transpose(0,2,1)@unique;truth=-logsumexp(normal_logpdf(r,np.zeros((128,5)),sigma))+np.log(128)
m=TemperedWishart(p,1,scales=(.4,),aux_power=.25,antithetic=True);nll=[]
for bank in range(12):
 _,d=m.update(j.asarray(cloud),j.asarray(r),jr.PRNGKey(310+bank));nll.append(float(d[-1]))
nll=np.array(nll);se=nll.std(ddof=1)/np.sqrt(len(nll));error=nll.mean()-truth
assert abs(error)<max(.015,4*se),(error,se)
# A frozen deterministic covariance under zero leverage must reduce to the
# exact Gaussian risk, under antithetic and balanced sampling too.
p0=dict(p,kappa=1e-12,rho=0.);m0=TemperedWishart(p0,1,scales=(.4,),antithetic=True);root=np.linalg.cholesky(c);y=np.zeros((128,6,5));y[:,:5]=root.T
mean,cov=m0.predict(j.asarray(y),jr.PRNGKey(330),1024);q,e=m0.risk(mean,cov,j.asarray([np.ones(5)/5]),j.array([.01,.025]))
from scipy.stats import norm
sd=np.sqrt(c.mean());qq=sd*norm.isf([.01,.025]);ee=sd*norm.pdf(norm.isf([.01,.025]))/np.array([.01,.025]);riskerr=max(abs(np.array(q)[0]-qq).max(),abs(np.array(e)[0]-ee).max());assert riskerr<1e-9
out=dict(passed=True,fully_adapted_reference_nll=float(truth),flattened_auxiliary_mean_nll=float(nll.mean()),flattened_auxiliary_nll_error=float(error),flattened_auxiliary_nll_seed_se=float(se),banks=12,particles=2048,antithetic_exact_gaussian_risk_error=float(riskerr),seconds=time.time()-t)
(ROOT/'wishart_completion/analysis/flattened_auxiliary_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
