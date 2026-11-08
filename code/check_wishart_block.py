"""Fixed-covariance block normalizer is an independent analytic target."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from wishart_tempered import TemperedWishart,jr,j
from wishart_block import BlockWishart
from wishart_model import normal_logpdf
ROOT=Path(__file__).resolve().parents[1];t=time.time()
c=.5*np.eye(5)+.5*np.ones((5,5));p=dict(C=c.tolist(),mu=[0.]*5,beta=6,kappa=1e-14,rho=0.,ito_scale=0.)
m=TemperedWishart(p,1);u=np.array(m.initial(jr.PRNGKey(400),128));cloud=np.tile(u,(16,1,1));sig=u.transpose(0,2,1)@u
obs=np.array([[-1.,-.8,-1.3,-.5,-.9],[.2,.3,.1,-.1,.5],[-2.,-1.,-.5,-1.2,-1.6]])
results=[]
for lag in [2,3]:
 model=BlockWishart(p,1,lag=lag,scales=(.4,));exact=-logsumexp(sum(normal_logpdf(r,np.zeros((128,5)),sig) for r in obs[:lag]))+np.log(128)
 values=[]
 for bank in range(10):
  _,d=model.update(j.asarray(cloud),j.asarray(obs[:lag]),jr.PRNGKey(410+bank));values.append(float(d[-1]));assert float(d[0])==1
 err=np.mean(values)-exact;se=np.std(values,ddof=1)/np.sqrt(len(values));assert abs(err)<max(.02,4*se),(lag,err,se)
 results.append(dict(lag=lag,exact_joint_nll=float(exact),mean_joint_nll=float(np.mean(values)),error=float(err),seed_se=float(se)))
out=dict(passed=True,frozen_joint_normalizers=results,seconds=time.time()-t)
(ROOT/'wishart_completion/analysis/block_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
