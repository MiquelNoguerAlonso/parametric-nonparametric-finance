"""Exact pathwise orthogonal invariance underlying factor canonicalisation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,json
from pathlib import Path
from wishart_tempered import TemperedWishart,jr,j
p=dict(C=(.5*np.eye(5)+.5*np.ones((5,5))).tolist(),mu=[0.]*5,beta=6,kappa=.2,rho=-.7,ito_scale=0.)
g=np.random.default_rng(340);m=TemperedWishart(p,8,moment_auxiliary=True);y=np.asarray(m.initial(jr.PRNGKey(340),128));o=np.linalg.qr(g.normal(size=(128,6,6)))[0]
z,e=m._noise(jr.PRNGKey(341),128);z=np.array(z);e=np.array(e);r=j.array([-2.,-1.5,-1.,-.5,-1.2]);yr=np.einsum('nij,njk->nik',o,y);zr=np.einsum('nij,nksjl->nksil',o,z)
a,la=m.proposal(j.asarray(y),r,j.asarray(z),j.asarray(e));b,lb=m.proposal(j.asarray(yr),r,j.asarray(zr),j.asarray(e));b_expected=np.einsum('nij,njk->nik',o,np.array(a));factor_error=float(np.max(abs(np.array(b)-b_expected)));ratio_error=float(np.max(abs(np.array(la)-np.array(lb))));assert factor_error<1e-7 and ratio_error<1e-6
from run_wishart_completion import canonicalise
cy=np.asarray(canonicalise(j.asarray(y)));cov_error=float(np.max(abs(cy.transpose(0,2,1)@cy-y.transpose(0,2,1)@y)));assert cov_error<1e-10
out=dict(passed=True,factor_equivariance_error=factor_error,log_ratio_invariance_error=ratio_error,canonical_covariance_error=cov_error)
Path('wishart_completion/analysis/gauge_checks.json').write_text(json.dumps(out,indent=2));print(out)
