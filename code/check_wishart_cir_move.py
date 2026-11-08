"""Independent checks of MH invariance and frozen-law characteristic functions."""
import json
from pathlib import Path
import numpy as np
from wishart_cir import initial

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'completion_audit/analysis'
p=json.loads((ROOT/'wishart_cir_extension/fits/crsp5_2025.json').read_text())['parameters']
rng=np.random.default_rng(910733)
# Exact finite-state detailed balance for the implemented independence-MH ratio.
proposal=np.array([.02,.1,.3,.58]); logweight=np.array([2.,-1.,.5,-.2])
target=proposal*np.exp(logweight);target/=target.sum()
accept=np.minimum(1,np.exp(logweight[None,:]-logweight[:,None]))
kernel=proposal[None,:]*accept
kernel[np.diag_indices(4)]+=1-kernel.sum(1)
balance=target[:,None]*kernel
assert np.max(abs(balance-balance.T))<1e-14
assert np.max(abs(target@kernel-target))<1e-14
# Independent stationary state simulation versus the analytic scalar-mixture CF.
n=150000;y,v=initial(p,n,rng);w=y.transpose(0,2,1)@y;c=np.array(p['C']);rows=[]
for u in [np.ones(5)/5,np.array([1.,-1.,0,0,0]),np.array([0,0,0,0,1.])]:
 q=u@c@u;eta=p['eta'];b=p['beta'];a=p['a']
 val=np.exp(-.5*(eta*np.einsum('i,nij,j->n',u,w,u)+(1-eta)*v*q))
 truth=(1+eta*q/b)**(-b/2)*(1+(1-eta)*q/(2*a))**(-a)
 se=val.std(ddof=1)/np.sqrt(n);z=abs(val.mean()-truth)/se
 assert z<6,(u,z)
 rows.append(dict(projection=u.tolist(),estimate=float(val.mean()),analytic=float(truth),se=float(se),z=float(z)))
result=dict(detailed_balance_max_error=float(np.max(abs(balance-balance.T))),stationary_max_error=float(np.max(abs(target@kernel-target))),frozen_characteristic_function=rows)
(OUT/'move_and_equivalence_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
