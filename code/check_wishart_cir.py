import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from wishart_cir import initial,step,constants,conditional,risk,windows_components,conditional_features
from wishart_model import features
p=dict(C=(.5*np.eye(5)+.5*np.ones((5,5))).tolist(),mu=[0]*5,beta=6,eta=.25,a=27/49,kappa_w=.05,kappa_v=3/252,rho=-.7,ito_scale=0)
rng=np.random.default_rng(4791);n=300000;y,v=initial(p,n,rng);yn,vn=step(y,v,p,rng,1)
w=y.transpose(0,2,1)@y;wn=yn.transpose(0,2,1)@yn
c,_,g=constants(p);assert np.linalg.eigvalsh(c-np.outer(g,g)).min()>0
checks={}
def test(name,values,target,tolerance=6):
    avg=float(np.mean(values));se=float(np.std(values,ddof=1)/np.sqrt(len(values)));z=abs(avg-target)/se
    checks[name]=dict(estimate=avg,target=float(target),se=se,z=z,pass_check=bool(z<tolerance));assert z<tolerance,(name,z)
test('stationary_v',v,1);test('stationary_v_second',v*v,1+1/p['a'])
e=np.exp(-p['kappa_v']);m=1+(v-1)*e;var=2*v*e*(1-e)/p['a']+(1-e)**2/p['a']
test('cir_conditional_mean',vn-m,0);test('cir_conditional_variance',(vn-m)**2-var,0)
test('wishart_conditional_mean',(wn[:,0,0]-(1+(w[:,0,0]-1)*np.exp(-p['kappa_w']))),0)
q=.25*w[:,0,0]+.75*v;k=3*(q*q).mean()/q.mean()**2-3
checks['frozen_kurtosis']=dict(estimate=float(k),target=3.125,pass_check=bool(abs(k-3.125)<.1));assert abs(k-3.125)<.1
sigma=p['eta']*w+(1-p['eta'])*v[:,None,None]*c
cross=(sigma[:,0,0]*sigma[:,1,1]+2*sigma[:,0,1]**2)/(c[0,0]*c[1,1])-sigma[:,0,0]/c[0,0]-sigma[:,1,1]/c[1,1]+1
rho0=c[0,1]/np.sqrt(c[0,0]*c[1,1])
test('frozen_cross_fourth_moment',cross,2*rho0*rho0+3.125/3*(1+2*rho0*rho0))
rr=risk(np.zeros(100),np.ones(100),.01);assert abs(rr['es']-norm.pdf(norm.ppf(.99))/.01)<1e-10
checks['normal_risk']=dict(**rr,pass_check=True)
# Conditional-moment implementation against actual residual Gaussian draws.
parts=windows_components(p,50000,456,8);analytic=conditional_features(p,parts,np.zeros(5)); draws=[]
for part in parts:
    mean,cov=conditional(p,part);draws.append(mean+np.einsum('nij,nj->ni',np.linalg.cholesky(cov),rng.standard_normal(mean.shape)))
windows=np.stack(draws,axis=2);emp=features(windows,c,np.zeros(5))
blocks=[features(b,c,np.zeros(5)) for b in np.array_split(windows,50)]
se=np.std(blocks,axis=0,ddof=1)/np.sqrt(50)
checks['conditional_moments']=dict(analytic=analytic.tolist(),sample=emp.tolist(),se=se.tolist(),pass_check=bool(np.all(abs(analytic-emp)<6*se)))
assert checks['conditional_moments']['pass_check']
path=Path(__file__).resolve().parents[1]/'wishart_cir_extension/analysis/kernel_checks.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
