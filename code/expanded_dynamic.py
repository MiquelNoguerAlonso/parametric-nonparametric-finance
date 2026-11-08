"""GARCH-t, conditional POT, and dynamic-covariance FHS for CRSP100."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import jax.numpy as jnp
from dynamic_benchmarks import fit_garch,unpack,variance,add
from expanded_experiment import risk_pairs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'expansion'
def main():
 p=argparse.ArgumentParser();p.add_argument('--years',nargs='+',type=int,default=list(range(2016,2026)));a=p.parse_args()
 for year in a.years:
  target=OUT/'results'/f'crsp100_{year}_dynamic.csv.gz'
  if target.exists():continue
  f=pd.read_csv(OUT/'data'/f'crsp100_{year}.csv',index_col=0,parse_dates=True);x=f.values*100;d=x.shape[1]
  cov=np.cov(x[:252],rowvar=False);cs=[];innov=[]
  for z in x:
   C=np.linalg.cholesky(cov+1e-8*np.eye(d));cs.append(C);innov.append(np.linalg.solve(C,z));cov=.94*cov+.06*np.outer(z,z)
  cs=np.asarray(cs);innov=np.asarray(innov)
  train=(f.index.year>=max(2006,year-11))&(f.index.year<=year-2);test=f.index.year==year;dates=f.index[test]
  weights={'equal':np.ones(d)/d,'top10':np.r_[np.ones(10)/10,np.zeros(d-10)],'largest':np.eye(d)[0]}
  start=np.where(train)[0][0];stop=np.where(test)[0][-1]+1;rows=[];pars=[]
  for pn,w in weights.items():
   loss=-x@w;p,initial,diag=fit_garch(loss[train]);mu,omega,arch,beta,nu=map(float,unpack(p))
   v=np.asarray(variance(p,jnp.asarray(loss[start:stop]),initial));sig=np.sqrt(v[np.where(test)[0]-start]);z=(loss[train]-mu)/np.sqrt(v[:train.sum()])
   u=float(np.quantile(z,.95));exc=z[z>u]-u;xi,_,scale=st.genpareto.fit(exc,floc=0);pu=len(exc)/len(z);assert xi<1
   pars.append(dict(year=year,portfolio=pn,mu=mu,omega=omega,arch=arch,beta=beta,nu=nu,initial=initial,evt_threshold=u,evt_xi=float(xi),evt_scale=float(scale),evt_mass=pu,**diag))
   proj=np.einsum('tij,i->tj',cs[test],w)
   for alpha in [.01,.025]:
    tq=st.t.isf(alpha,nu);c=np.sqrt((nu-2)/nu)
    add(rows,dates,pn,'GARCH-t',loss[test],mu+sig*c*tq,mu+sig*c*(nu+tq*tq)/(nu-1)*st.t.pdf(tq,nu)/alpha,alpha,year)
    qe=u+scale/xi*((pu/alpha)**xi-1) if abs(xi)>1e-8 else u+scale*np.log(pu/alpha);ee=(qe+scale-xi*u)/(1-xi)
    add(rows,dates,pn,'GARCH-EVT',loss[test],mu+sig*qe,mu+sig*ee,alpha,year)
    q,e=risk_pairs(innov[train],proj,alpha);add(rows,dates,pn,'EWMAcov-FHS',loss[test],q,e,alpha,year)
  pd.DataFrame(rows).to_csv(target,index=False);pd.DataFrame(pars).to_csv(target.with_name(f'crsp100_{year}_dynamic_parameters.csv'),index=False)
  print('Dynamic100',year,flush=True)
if __name__=='__main__':
 import fcntl
 with open(OUT/'dynamic100.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  main()
