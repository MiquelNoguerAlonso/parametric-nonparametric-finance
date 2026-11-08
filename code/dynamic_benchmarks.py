"""Additional, retrospectively specified CRSP benchmarks; no test-data tuning.

Annual parameter estimation uses exactly the original estimation dates. The
intervening validation year only updates states. All VaR/ES calculations here
are analytic or use the full empirical innovation law, without scenario MC.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import minimize
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import gammaln
from experiment import es

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'revision';OUT.mkdir(exist_ok=True)

def unpack(p):
    z=jax.nn.softmax(jnp.array([p[2],p[3],0.]))*.9999
    return p[0],jnp.exp(p[1]),z[0],z[1],p[4]

def variance(p,x,initial):
    mu,omega,a,b,nu=unpack(p)
    def step(v,xt):return omega+a*(xt-mu)**2+b*v,v
    return jax.lax.scan(step,initial,x)[1]

@jax.jit
def objective(p,x,initial):
    mu,omega,a,b,nu=unpack(p);v=variance(p,x,initial)
    # Student innovations have variance one, not scale one.
    lp=gammaln((nu+1)/2)-gammaln(nu/2)-.5*jnp.log((nu-2)*jnp.pi*v)
    lp-=(nu+1)/2*jnp.log1p((x-mu)**2/((nu-2)*v))
    return -lp.mean()
value_grad=jax.jit(jax.value_and_grad(objective))

def fit_garch(x):
    initial=float(np.var(x[:min(252,len(x))]));v=float(np.var(x))
    fits=[]
    for a,b,nu in [(.05,.9,8.),(.1,.85,5.),(.03,.95,15.)]:
        p=[np.mean(x),np.log(v*(1-a-b)),np.log(a/(1-a-b)),np.log(b/(1-a-b)),nu]
        fun=lambda p:tuple(np.asarray(q) for q in value_grad(jnp.asarray(p),jnp.asarray(x),initial))
        f=minimize(fun,p,jac=True,method='L-BFGS-B',bounds=[(None,None),(-20,10),(-14,14),(-14,14),(2.05,200)],options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7})
        fits.append(f)
    valid=[f for f in fits if f.success and np.isfinite(f.fun)]
    if not valid:raise RuntimeError([f.message for f in fits])
    best=min(valid,key=lambda f:f.fun)
    return best.x,initial,{'success':bool(best.success),'objective':float(best.fun),'iterations':int(best.nit),'message':str(best.message)}

def add(rows,dates,pn,name,actual,q,e,alpha,fold):
    q=np.broadcast_to(q,len(actual));e=np.broadcast_to(e,len(actual))
    if not (np.all(e>0) and np.all(e>=q)):raise ValueError((name,pn,fold))
    for i,day in enumerate(dates):
        rows.append(dict(date=str(day.date()),fold=fold,seed=0,model=name,portfolio=pn,alpha=alpha,loss=actual[i],var=q[i],es=e[i],hit=int(actual[i]>q[i]),fz0=np.log(e[i])+q[i]/e[i]-1+max(actual[i]-q[i],0)/(alpha*e[i]),es_residual=e[i]-q[i]-max(actual[i]-q[i],0)/alpha))

def main():
    f=pd.read_csv(ROOT/'data/crsp_size_portfolios.csv',index_col=0,parse_dates=True);x=f.values*100
    # Predictable full covariance, followed by triangular whitening.
    cov=np.cov(x[:252],rowvar=False);Cs=[];innov=[]
    for xt in x:
        C=np.linalg.cholesky(cov+1e-10*np.eye(5));Cs.append(C);innov.append(np.linalg.solve(C,xt))
        cov=.94*cov+.06*np.outer(xt,xt)
    Cs=np.array(Cs);innov=np.array(innov);rows=[];params=[]
    weights={'equal':np.ones(5)/5,'small':np.eye(5)[0],'large':np.eye(5)[-1]}
    for year in range(2016,2026):
        train=(f.index.year>=year-11)&(f.index.year<=year-2)&(f.index.year>=2006)
        test=f.index.year==year;dates=f.index[test]
        start=np.where(train)[0][0];stop=np.where(test)[0][-1]+1
        for pn,w in weights.items():
            loss=-x@w;p,initial,diag=fit_garch(loss[train])
            mu,omega,a,b,nu=map(float,unpack(p));v=np.asarray(variance(p,jnp.asarray(loss[start:stop]),initial))
            sigma=np.sqrt(v[np.where(test)[0]-start]);vt=np.sqrt(v[:train.sum()]);z=(loss[train]-mu)/vt
            # McNeil--Frey-style POT stage on GARCH-t standardized losses.
            u=float(np.quantile(z,.95));excess=z[z>u]-u;xi,_,scale=st.genpareto.fit(excess,floc=0)
            if xi>=1:raise ValueError('EVT fitted infinite mean')
            p_u=len(excess)/len(z)
            params.append(dict(year=year,portfolio=pn,mu=mu,omega=omega,arch=a,beta=b,nu=nu,initial=initial,evt_threshold=u,evt_mass=p_u,evt_xi=float(xi),evt_scale=float(scale),n_train=int(train.sum()),**diag))
            # Full empirical innovation distribution: no resampling error.
            empirical=-innov[train]@(np.einsum('tij,i->tj',Cs[test],w)).T
            for alpha in [.01,.025]:
                tq=st.t.isf(alpha,nu);c=np.sqrt((nu-2)/nu)
                q=mu+sigma*c*tq;e=mu+sigma*c*(nu+tq*tq)/(nu-1)*st.t.pdf(tq,nu)/alpha
                add(rows,dates,pn,'GARCH-t',loss[test],q,e,alpha,year)
                ze=u+scale/xi*((p_u/alpha)**xi-1) if abs(xi)>1e-8 else u+scale*np.log(p_u/alpha)
                ee=(ze+scale-xi*u)/(1-xi)
                add(rows,dates,pn,'GARCH-EVT',loss[test],mu+sigma*ze,mu+sigma*ee,alpha,year)
                q=np.quantile(empirical,1-alpha,axis=0,method='inverted_cdf');e=np.array([es(col,alpha) for col in empirical.T])
                add(rows,dates,pn,'EWMAcov-FHS',loss[test],q,e,alpha,year)
        print('Completed dynamic benchmarks',year,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'dynamic_forecasts.csv.gz',index=False)
    pd.DataFrame(params).to_csv(OUT/'dynamic_parameters.csv',index=False)
    (OUT/'dynamic_protocol.json').write_text(json.dumps(dict(garch='constant mean, stationary GARCH(1,1), standardized Student innovations',starts=3,df_bounds=[2.05,200],covariance_lambda=.94,evt_tail_fraction=.05,selection='fixed specification; training only',extra_parameter_fits=30),indent=2))

if __name__=='__main__':main()
