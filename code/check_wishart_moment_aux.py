"""Moment proposal checked against direct prior simulation, not its own weights."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from wishart_tempered import TemperedWishart,jr,j,logpdf
ROOT=Path(__file__).resolve().parents[1];t=time.time();rng=np.random.default_rng(4401)
c=.5*np.eye(5)+.5*np.ones((5,5));p=dict(C=c.tolist(),mu=[0.]*5,beta=6,kappa=.2,rho=-.7,ito_scale=0.)
base=TemperedWishart(p,8);start=np.asarray(base.initial(jr.PRNGKey(450),1))[0];fixed=np.repeat(start[None],128,axis=0);sig=start.T@start;r=np.array([-.4,-.6,-.3,-.5,-.2])
m=TemperedWishart(p,8,scales=(.4,),moment_auxiliary=True)
reference=[];guided=[];covs=[];means=[]
for b in range(8):
 mean,cov=base.predict(j.asarray(fixed),jr.PRNGKey(460+b),16384)
 reference.append(float(j.exp(logpdf(j.asarray(r),mean,cov)).mean()))
 covs.append(np.asarray(cov).mean(0)+np.asarray(mean).T@np.asarray(mean)/len(mean));means.append(np.asarray(mean).mean(0))
 y=np.repeat(start[None],16384,axis=0);z=rng.normal(size=(16384,2,8,6,5));eps=rng.normal(size=(16384,5));_,lr=m.proposal(j.asarray(y),j.asarray(r),j.asarray(z),j.asarray(eps))
 average=c+m.average_decay*(sig-c);l0=float(logpdf(j.asarray(r),j.zeros(5),j.asarray(average)))
 guided.append(float(j.exp(lr+l0).mean()))
reference=np.array(reference);guided=np.array(guided);se=np.sqrt(reference.var(ddof=1)/8+guided.var(ddof=1)/8);err=guided.mean()-reference.mean();assert abs(err)<4*se,(err,se)
expected=c+m.average_decay*(sig-c);cov_error=np.max(abs(np.mean(covs,0)-expected));assert cov_error<.07
# Exact one-step limiting case remains fully adapted.
m1=TemperedWishart(p,1,scales=(.4,),moment_auxiliary=True);y=m1.initial(jr.PRNGKey(480),1024);_,d=m1.update(y,j.asarray(r),jr.PRNGKey(481));sig1=np.asarray(y).transpose(0,2,1)@np.asarray(y);true=-float(logsumexp(np.asarray(logpdf(j.asarray(r),j.zeros((1024,5)),j.asarray(sig1)))))+np.log(1024);limerr=abs(float(d[-1])-true);assert limerr<1e-8
out=dict(passed=True,prior_density=float(reference.mean()),guided_density=float(guided.mean()),density_difference=float(err),combined_mc_se=float(se),z=float(err/se),conditional_return_covariance_max_error=float(cov_error),fully_adapted_one_step_nll_error=float(limerr),mean_decay=float(m.average_decay),seconds=time.time()-t)
(ROOT/'wishart_completion/analysis/moment_aux_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
