"""Contracts that do not mirror training code: empirical ES, selection, chronology."""
from pathlib import Path
import json
import numpy as np
import scipy.stats as st
from expanded_experiment import risk_pairs,FZValidation
from experiment import es
from models import Neural
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,condition,detail):
 checks.append(dict(check=name,passed=bool(condition),detail=detail))
 if not condition:raise AssertionError(name)
def main():
 rng=np.random.default_rng(74);X=rng.standard_t(5,(2256,3));projections=rng.normal(size=(8,3));alpha=.01
 q,e=risk_pairs(X,projections,alpha)
 # Independent Rockafellar empirical formula using NumPy quantiles.
 loss=-X@projections.T;indq=np.quantile(loss,1-alpha,axis=0,method='inverted_cdf');inde=indq+np.maximum(loss-indq,0).mean(0)/alpha
 check('Exact empirical ES including fractional tail mass',np.allclose(e,inde,atol=1e-11) and np.allclose(q,indq),float(np.max(abs(e-inde))))
 validation=np.linspace(-1,4,8);v=FZValidation(validation,projections);manual=np.mean(np.log(inde)+indq/inde-1+np.maximum(validation-indq,0)/(alpha*inde))
 check('Common FZ0 selection accounting',abs(v(X)-manual)<1e-10,float(v(X)))
 V=rng.normal(size=(100,2));X=rng.normal(size=(250,2));weights=np.ones(2)/2;callback=FZValidation(-V@weights,weights)
 # One checkpoint provides a direct audit that the common callback is used.
 for kind in ['flow','vae','gan']:
  n=Neural(kind,2,seed=19,steps=250).fit(X,V,risk_validation=callback)
  import jax
  sampled=np.asarray(n.sample_std(n.p,jax.random.PRNGKey(424242),20000))*n.scale+n.loc
  expected=callback(sampled)
  check(kind+' saved trace equals validation FZ0',abs(expected-n.trace[0][1])<2e-4,dict(trace=n.trace[0][1],recomputed=expected))
 a=json.loads((ROOT/'expansion/data/crsp100_audit.json').read_text())
 check('Annual universe uses training-end information',all(int(z['universe_date'][:4])==z['test_year']-2 and len(z['permnos'])==len(set(z['permnos']))==100 for z in a['universes']),len(a['universes']))
 check('No test-availability screen',any(z['missing_test_cells']>0 for z in a['universes']) and all(z['train_min_fraction']>=.98 and z['minimum_pre_estimation_observations']>=252 for z in a['universes']),'missing test observations retained and explicitly masked')
 (ROOT/'expansion/checks_preflight.json').write_text(json.dumps(checks,indent=2));print(len(checks),'checks passed')
if __name__=='__main__':main()
