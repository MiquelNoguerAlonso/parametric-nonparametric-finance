"""Additive Wishart--CIR model; exact state transitions, integrated Gaussian returns.

The scalar variance driver also drives one return factor. Conditional on the
state paths the remaining return noise is Gaussian. Trapezoidal quadrature is
the only time discretization; exact transitions do NOT imply exact returns.
"""
import itertools
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, logsumexp
from wishart_model import symmetric_root, features, normal_logpdf, systematic


def constants(p):
    c=np.asarray(p['C']); w=np.ones(len(c))/len(c)
    g=p['rho']*c@w/np.sqrt(w@c@w)
    return c, symmetric_root(c)/np.sqrt(p['beta']), g


def initial(p,n,rng):
    c,root,g=constants(p)
    return rng.standard_normal((n,p['beta'],len(c)))@root, rng.gamma(p['a'],1/p['a'],n)


def step(y,v,p,rng,dt):
    c,root,g=constants(p)
    decay=np.exp(-p['kappa_w']*dt/2)
    yn=decay*y+np.sqrt(-np.expm1(-p['kappa_w']*dt))*(rng.standard_normal(y.shape)@root)
    ev=np.exp(-p['kappa_v']*dt); scale=-np.expm1(-p['kappa_v']*dt)/(2*p['a'])
    vn=scale*rng.noncentral_chisquare(2*p['a'],v*ev/scale)
    return yn,vn


def integrals(y,v,p,rng,steps=8,levels=None):
    """One day. Nested coarser grids share exact endpoints and CIR variates."""
    levels=[steps] if levels is None else levels
    assert all(steps%s==0 for s in levels)
    c,_,_=constants(p); corr=c/np.sqrt(np.outer(np.diag(c),np.diag(c)))
    w=y.transpose(0,2,1)@y; v0=v.copy()
    store={s:[np.zeros_like(w),np.zeros_like(v),np.zeros_like(w),w.copy(),v.copy()] for s in levels}
    for j in range(1,steps+1):
        y,v=step(y,v,p,rng,1/steps); w=y.transpose(0,2,1)@y
        for s in levels:
            if j%(steps//s):continue
            iw,iv,iff,wp,vp=store[s]
            iw+=(wp+w)/(2*s); iv+=(vp+v)/(2*s)
            dp=np.sqrt(np.diagonal(wp,axis1=1,axis2=2)); dn=np.sqrt(np.diagonal(w,axis1=1,axis2=2))
            iff+=(dp[:,:,None]*corr*dp[:,None,:]+dn[:,:,None]*corr*dn[:,None,:])/(2*s)
            store[s][3:]=[w.copy(),v.copy()]
    return y,v,{s:(r[0],r[1],r[2],v-v0) for s,r in store.items()}


def conditional(p,integ,ablation='full'):
    iw,iv,iff,dv=integ; c,_,g=constants(p); eta=p['eta']
    if ablation=='fixed_component_correlation':iw=iff
    if ablation=='no_leverage':g=np.zeros_like(g)
    total=eta*iw+(1-eta)*iv[:,None,None]*c
    xi=np.sqrt(2*p['kappa_v']/p['a'])
    mean=np.asarray(p['mu'])-.5*p.get('ito_scale',0)*np.diagonal(total,axis1=1,axis2=2)
    mean=mean+np.sqrt(1-eta)*((dv-p['kappa_v']+p['kappa_v']*iv)/xi)[:,None]*g
    cov=total-(1-eta)*iv[:,None,None]*np.outer(g,g)
    return mean,cov


def windows_components(p,n,seed,steps=8):
    rng=np.random.default_rng(seed); y,v=initial(p,n,rng); result=[]
    for _ in range(5):
        y,v,r=integrals(y,v,p,rng,steps);result.append(r[steps])
    return result


def conditional_features(p,parts,daily_mean):
    """Exact conditional moments; integrate all residual Gaussian draw noise."""
    mv=[conditional(p,q) for q in parts]; c=np.asarray(p['C']); d=len(c)
    m=np.stack([r[0] for r in mv],axis=2)-daily_mean[None,:,None]
    cov=np.stack([r[1] for r in mv],axis=3)/np.sqrt(np.outer(np.diag(c),np.diag(c)))[None,:,:,None]
    m=m/np.sqrt(np.diag(c))[None,:,None]
    var=np.stack([cov[:,i,i,:] for i in range(d)],axis=1)
    sq=m*m+var-1
    out=[np.mean(m**4+6*m*m*var+3*var*var)-3]
    out += [np.mean(sq[:,:,:-lag]*sq[:,:,lag:]) for lag in range(1,5)]
    out += [np.mean(m[:,:,:-lag]*sq[:,:,lag:]) for lag in range(1,5)]
    i,j=np.triu_indices(d,1)
    out += [np.mean(sq[:,i,:]*sq[:,j,:]+2*cov[:,i,j,:]**2+4*m[:,i,:]*m[:,j,:]*cov[:,i,j,:])]
    return np.asarray(out)


def fit(windows,seed,ito_scale=0,daily=None,n=8192,steps=8):
    d=windows.shape[1]
    daily=windows.transpose(0,2,1).reshape(-1,d) if daily is None else daily
    mean=daily.mean(0); c=np.cov(daily,rowvar=False); c=.99*c+.01*np.diag(np.diag(c))
    obs=features(windows,c,mean); scales=np.array([3,1,1,1,1,.1,.1,.1,.1,1])
    candidates=[]
    for eta,kw,kv in itertools.product([.1,.25,.5],[.01,.05,.2],[.005,3/252,.05]):
        a=float(np.clip(3*(1-eta)**2/max(obs[0]-6*eta**2/(d+1),.01),.05,20))
        p=dict(C=c.tolist(),mu=(mean+.5*ito_scale*np.diag(c)).tolist(),beta=d+1,
               eta=eta,a=a,kappa_w=kw,kappa_v=kv,rho=0.,ito_scale=ito_scale)
        parts=windows_components(p,n,seed,steps)
        for rho in [-.7,0.,.7]:
            p['rho']=rho; mom=conditional_features(p,parts,mean)
            obj=float(np.mean(((mom-obs)/scales)**2))
            candidates.append(dict(parameters=p.copy(),objective=obj,moments=mom.tolist()))
    best=min(candidates,key=lambda x:x['objective'])
    return best['parameters'],dict(candidates=candidates,observed_moments=obs.tolist(),
        simulated_moments=best['moments'],objective=best['objective'],moment_scales=scales.tolist(),
        fit_scenarios=n,substeps=steps,seed=seed,n_windows=len(windows),n_daily=len(daily),
        method='training-only fixed-grid SMM with conditional Gaussian moments; a anchored to pooled training kurtosis')


def risk(loc,sd,alpha):
    q=brentq(lambda q:ndtr((loc-q)/sd).mean()-alpha,float(np.min(loc-15*sd)),float(np.max(loc+15*sd)))
    z=(q-loc)/sd; prob=ndtr(-z); ep=loc*prob+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
    es=float(ep.mean()/alpha); influence=(ep-q*prob)/alpha+q-es
    return dict(var=float(q),es=es,mc_se=float(influence.std(ddof=1)/np.sqrt(len(loc))))


def project(m,c,w):
    return -m@w,np.sqrt(np.einsum('i,nij,j->n',w,c,w))


def update(y,v,p,x,rng,steps=8,guided=False):
    """Auxiliary PF with exact likelihood correction, no observation in forecasts."""
    c,_,_=constants(p); n=len(v); w=y.transpose(0,2,1)@y
    fw=-np.expm1(-p['kappa_w'])/p['kappa_w']; fv=-np.expm1(-p['kappa_v'])/p['kappa_v']
    sig=p['eta']*(c+fw*(w-c))+(1-p['eta'])*(1+fv*(v-1))[:,None,None]*c
    mu=np.asarray(p['mu'])-.5*p.get('ito_scale',0)*np.diagonal(sig,axis1=1,axis2=2)
    aux=normal_logpdf(x,mu,sig); la=logsumexp(aux); prob=np.exp(aux-la)
    ancestors=systematic(prob,n,rng)
    if guided:
        ya,va=y[ancestors],v[ancestors]
        _,_,g=constants(p)
        xi=np.sqrt(2*p['kappa_v']/p['a'])
        displacement=xi*np.sqrt(1-p['eta'])*np.einsum('i,ni->n',g,np.linalg.solve(sig[ancestors],(x-mu[ancestors])[...,None])[...,0])
        delta=np.clip(np.log1p(np.maximum(displacement,-.8)),-1.,1.5)
        yn,vn,integ,correction=guided_integrals(ya,va,p,rng,steps,delta)
    else:
        yn,vn,r=integrals(y[ancestors],v[ancestors],p,rng,steps);integ=r[steps];correction=0.
    m,cv=conditional(p,integ); lw=normal_logpdf(x,m,cv)-aux[ancestors]+correction
    lw_norm=logsumexp(lw); weights=np.exp(lw-lw_norm)
    pick=systematic(weights,n,rng)
    return yn[pick],vn[pick],dict(ess=float(1/np.sum(weights**2)),
        parent_ess=float(1/np.sum(prob**2)),nll=float(-la+np.log(n)-lw_norm+np.log(n)))


def guided_integrals(y,v,p,rng,steps,delta):
    """Defensive mixture of prior paths and exact exponential-tilt CIR paths.

    Tilted ncx2 transition: c*s*chi2(df, lambda*s), s=exp(delta*dt/2).
    log(q/p) = ((s-1)/(2*c*s))*v_next - a*log(s) - lambda*(s-1)/2.
    One mixture indicator per WHOLE daily path, so corrections mix PATH densities.
    """
    c,root,g=constants(p);n=len(v);v0=v.copy();w=y.transpose(0,2,1)@y
    iw=np.zeros_like(w);iv=np.zeros_like(v);dt=1/steps
    s=np.exp(delta*dt/2);tilted=rng.random(n)<.5;ss=np.where(tilted,s,1.)
    cv=-np.expm1(-p['kappa_v']*dt)/(2*p['a']);dec=np.exp(-p['kappa_v']*dt)
    ou=np.exp(-p['kappa_w']*dt/2);ousd=np.sqrt(-np.expm1(-p['kappa_w']*dt))
    logratio=np.zeros(n)
    for _ in range(steps):
        lam=v*dec/cv
        vn=cv*ss*rng.noncentral_chisquare(2*p['a'],lam*ss)
        logratio+=(s-1)*vn/(2*cv*s)-p['a']*np.log(s)-lam*(s-1)/2
        yn=ou*y+ousd*(rng.standard_normal(y.shape)@root);wn=yn.transpose(0,2,1)@yn
        iw+=(w+wn)*dt/2;iv+=(v+vn)*dt/2;y,v,w=yn,vn,wn
    correction=-np.logaddexp(np.log(.5),np.log(.5)+logratio)
    return y,v,(iw,iv,iw,v-v0),correction
