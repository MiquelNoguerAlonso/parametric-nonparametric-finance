"""Tempered correction of the existing Wishart auxiliary proposal.

No parameter or diffusion-law change. The reference measure is categorical
ancestor weights times independent standard normals. pCN and independent
ancestor refresh preserve that reference; MH uses only the tempered density
correction. Returned equal-weight factors approximate the filtering posterior.
"""
from functools import partial
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
from jax import lax, random as jr
from jax.scipy.special import logsumexp, ndtr


def roots(s):
    v,u=j.linalg.eigh(s)
    # No eigenvalue clipping/projection: loss of rank is a numerical failure.
    return (u*j.sqrt(v)[...,None,:])@j.swapaxes(u,-1,-2),(u/j.sqrt(v)[...,None,:])@j.swapaxes(u,-1,-2)


def logpdf(r,m,c):
    l=j.linalg.cholesky(c); z=j.linalg.solve(l,(r-m)[...,None])[...,0]
    return -.5*(m.shape[-1]*j.log(2*j.pi)+2*j.log(j.diagonal(l,axis1=-2,axis2=-1)).sum(-1)+(z*z).sum(-1))


def resample(prob,key):
    n=prob.shape[0]; u=(jr.uniform(key,dtype=j.float64)+j.arange(n))/n
    c=j.cumsum(prob).at[-1].set(1.)
    return j.searchsorted(c,u,side='right')


class TemperedWishart:
    def __init__(self,par,substeps=8,ess_fraction=.7,scales=(.2,.4,.6),max_stages=80,antithetic=False,defensive=0.,moment_auxiliary=False,aux_power=1.,gauge_paths=False):
        self.gauge_paths=gauge_paths; self.aux_power=aux_power
        if defensive and aux_power!=1.:raise ValueError("Defensive and flattened auxiliary are separate proposals")
        self.par=par; self.substeps=substeps; self.ess_fraction=ess_fraction; self.antithetic=antithetic; self.defensive=defensive; self.moment_auxiliary=moment_auxiliary
        self.scales=tuple(scales); self.max_stages=max_stages
        c=j.asarray(par['C']);self.d=c.shape[0];self.beta=int(par['beta'])
        self.c=c; self.croot=roots(c)[0]; self.mu=j.asarray(par['mu']);self.k=float(par['kappa']);self.rho=float(par['rho'])
        self.ito=float(par.get('ito_scale',0.));self.dt=1./substeps
        self.average_decay=-np.expm1(-self.k)/(substeps*(-np.expm1(-self.k/substeps)))
        self.q=j.sqrt(self.k/self.beta)*self.croot
        self.a=np.exp(-self.k*self.dt/2)
        self.ouv=-np.expm1(-self.k*self.dt)/self.k
        self.cross=-np.expm1(-self.k*self.dt/2)/(self.k/2)
        self.ou_res=np.sqrt(max(0.,self.ouv-self.cross**2/self.dt))
        self.initial=jax.jit(self._initial,static_argnums=(1,))
        self.proposal=jax.jit(self._proposal)
        self.update=jax.jit(self._update)
        self.predict=jax.jit(self._predict,static_argnums=(2,))
        self.risk=jax.jit(self._risk)

    def _initial(self,key,n):
        return jr.normal(key,(n,self.beta,self.d),dtype=j.float64)@self.croot/j.sqrt(self.beta)

    def _noise(self,key,n):
        k1,k2=jr.split(key)
        return (jr.normal(k1,(n,2,self.substeps,self.beta,self.d),dtype=j.float64),
                jr.normal(k2,(n,self.d),dtype=j.float64))

    def _proposal(self,start,observed,z,eps,guided=None,logmeanaux=None):
        """Deterministic push-forward of the frozen conditional proposal."""
        sigma=start.swapaxes(1,2)@start;root,invroot=roots(sigma)
        if self.gauge_paths:start=j.zeros_like(start).at[:,:self.d,:].set(root)
        average=self.c+self.average_decay*(sigma-self.c) if self.moment_auxiliary else sigma
        residual=average-self.rho**2*sigma
        drift0=self.mu-.5*self.ito*j.diagonal(average,axis1=1,axis2=2)
        projection=invroot@start.swapaxes(1,2)
        db=j.sqrt(self.dt)*z[:,0]
        frozen_dw=projection@db.sum(1)
        if self.moment_auxiliary:
            resroot=roots(residual)[0]
            pseudo=drift0+self.rho*j.einsum('nij,nj->ni',root,j.diagonal(frozen_dw,axis1=1,axis2=2))+j.einsum('nij,nj->ni',resroot,eps)
            gain_target=j.einsum('nij,nj->ni',root,j.linalg.solve(average,(observed-pseudo)[...,None])[...,0])
        else:
            pseudo=self.rho*j.diagonal(frozen_dw,axis1=1,axis2=2)+j.sqrt(1-self.rho**2)*eps
            target=j.einsum('nij,nj->ni',invroot,observed-drift0)
            gain_target=target-pseudo
        shift=self.dt*self.rho*projection.swapaxes(1,2)*gain_target[:,None,:]
        db=db+shift[:,None]*(1. if guided is None else guided[:,None,None,None])
        dw0=projection@db.sum(1)
        frozen_mean=drift0+self.rho*j.einsum('nij,nj->ni',root,j.diagonal(dw0,axis1=1,axis2=2))
        frozen_log=logpdf(observed,frozen_mean,residual)
        def step(carry,drivers):
            y,mean,cov=carry;delta,eta=drivers
            sig=y.swapaxes(1,2)@y;rt,irt=roots(sig)
            if self.gauge_paths:y=j.zeros_like(y).at[:,:self.d,:].set(rt)
            dw=irt@(y.swapaxes(1,2)@delta)
            mean=mean+(self.mu-.5*self.ito*j.diagonal(sig,axis1=1,axis2=2))*self.dt+self.rho*j.einsum('nij,nj->ni',rt,j.diagonal(dw,axis1=1,axis2=2))
            cov=cov+(1-self.rho**2)*sig*self.dt
            yn=self.a*y+((self.cross/self.dt)*delta+self.ou_res*eta)@self.q
            return (yn,mean,cov),None
        n=start.shape[0]
        (end,mean,cov),_=lax.scan(step,(start,j.zeros((n,self.d)),j.zeros((n,self.d,self.d))),
                               (db.swapaxes(0,1),z[:,1].swapaxes(0,1)))
        actual=logpdf(observed,mean,cov)
        if self.defensive:
            logden=j.logaddexp(j.log(self.defensive),j.log1p(-self.defensive)+frozen_log-logmeanaux)
            return end,actual-logden
        return end,actual-frozen_log

    def _update(self,cloud,observed,key):
        n=cloud.shape[0];sigma=cloud.swapaxes(1,2)@cloud
        average=self.c+self.average_decay*(sigma-self.c) if self.moment_auxiliary else sigma
        drift=self.mu-.5*self.ito*j.diagonal(average,axis1=1,axis2=2)
        aux=logpdf(observed,drift,average);normaux=logsumexp(self.aux_power*aux)
        logmeanaux=normaux-j.log(n);ap=j.exp(self.aux_power*aux-normaux);pess=1/j.sum(ap**2)
        def draw_ancestor(key):
            if not self.defensive:
                ii=j.searchsorted(j.cumsum(ap).at[-1].set(1.),jr.uniform(key,(n,),dtype=j.float64),side='right')
                return ii,j.ones(n,dtype=bool)
            kg,ki=jr.split(key);guided=jr.uniform(kg,(n,),dtype=j.float64)>=self.defensive
            uu=jr.uniform(ki,(n,),dtype=j.float64)
            ia=j.searchsorted(j.cumsum(ap).at[-1].set(1.),uu,side='right')
            ip=j.minimum((uu*n).astype(ia.dtype),n-1)
            return j.where(guided,ia,ip),guided
        key,k1,k2=jr.split(key,3)
        if self.defensive:idx,guided=draw_ancestor(k1)
        else:idx,guided=resample(ap,k1),j.ones(n,dtype=bool)
        z,eps=self._noise(k2,n)
        def evaluate(idx,guided,z,eps):
            end,ratio=self._proposal(cloud[idx],observed,z,eps,guided,logmeanaux)
            return end,ratio+(1-self.aux_power)*aux[idx]
        end,lr=evaluate(idx,guided,z,eps)
        rawess=1/j.sum(jax.nn.softmax(lr)**2)
        state=(j.array(0.),idx,guided,z,eps,end,lr,key,j.array(0),j.array(0.),j.array(float(n)),j.array(0.),j.array(0.))
        def cond(s):return (s[0]<1.-1e-12)&(s[8]<self.max_stages)
        def body(s):
            gamma,idx,guided,z,eps,end,lr,key,st,logz,miness,acc,attempt=s
            def ess(delta):return 1/j.sum(jax.nn.softmax(delta*lr)**2)
            def bisect(_,bounds):
                lo,hi=bounds;mid=(lo+hi)/2;ok=ess(mid)>=self.ess_fraction*n
                return j.where(ok,mid,lo),j.where(ok,hi,mid)
            lo,_=lax.fori_loop(0,32,bisect,(j.array(0.),1.-gamma))
            delta=j.where(ess(1.-gamma)>=self.ess_fraction*n,1.-gamma,lo)
            gamma=gamma+delta;lw=delta*lr;logz=logz+logsumexp(lw)-j.log(n)
            prob=jax.nn.softmax(lw);miness=j.minimum(miness,1/j.sum(prob**2))
            key,kr=jr.split(key);sel=resample(prob,kr)
            idx,guided,z,eps,end,lr=idx[sel],guided[sel],z[sel],eps[sel],end[sel],lr[sel]
            for scale in self.scales:
                key,kn,ku=jr.split(key,3);zn,en=self._noise(kn,n)
                zp=j.sqrt(1-scale**2)*z+scale*zn;ep=j.sqrt(1-scale**2)*eps+scale*en
                yp,lp=evaluate(idx,guided,zp,ep)
                accept=j.log(jr.uniform(ku,(n,),dtype=j.float64))<gamma*(lp-lr)
                z=j.where(accept[:,None,None,None,None],zp,z);eps=j.where(accept[:,None],ep,eps)
                end=j.where(accept[:,None,None],yp,end);lr=j.where(accept,lp,lr)
                acc+=accept.sum();attempt+=n
            # Joint component/ancestor refresh preserves their reference law.
            key,ki,ku=jr.split(key,3);ip,gp=draw_ancestor(ki)
            yp,lp=evaluate(ip,gp,z,eps)
            accept=j.log(jr.uniform(ku,(n,),dtype=j.float64))<gamma*(lp-lr)
            idx=j.where(accept,ip,idx);guided=j.where(accept,gp,guided)
            end=j.where(accept[:,None,None],yp,end);lr=j.where(accept,lp,lr)
            acc+=accept.sum();attempt+=n
            return gamma,idx,guided,z,eps,end,lr,key,st+1,logz,miness,acc,attempt
        out=lax.while_loop(cond,body,state)
        gamma,idx,guided,z,eps,end,lr,key,st,logz,miness,acc,attempt=out
        nll=-logz if self.defensive else -logmeanaux-logz
        return end,j.stack([gamma,st,miness,pess,rawess,acc/j.maximum(attempt,1),nll])

    def _predict(self,cloud,key,n_predictive):
        key,ki=jr.split(key);k1,k2=jr.split(key)
        if self.antithetic:
            if n_predictive % (2*cloud.shape[0]):
                raise ValueError('Antithetic prediction requires an even multiple of particles')
            half=n_predictive//2;idx=j.arange(half)%cloud.shape[0]
            y=j.concatenate([cloud[idx],cloud[idx]],axis=0)
            za=jr.normal(k1,(self.substeps,half,self.beta,self.d),dtype=j.float64)
            zb=jr.normal(k2,za.shape,dtype=j.float64)
            z=j.concatenate([za,-za],axis=1);zr=j.concatenate([zb,-zb],axis=1)
        else:
            idx=jr.randint(ki,(n_predictive,),0,cloud.shape[0]);y=cloud[idx]
            z=jr.normal(k1,(self.substeps,n_predictive,self.beta,self.d),dtype=j.float64)
            zr=jr.normal(k2,z.shape,dtype=j.float64)
        ou_sd=j.sqrt(self.ouv);bc=self.cross/ou_sd;br=j.sqrt(j.maximum(0.,self.dt-bc**2))
        def step(carry,noise):
            y,mean,cov=carry;z,zr=noise;sig=y.swapaxes(1,2)@y;rt,irt=roots(sig)
            if self.gauge_paths:y=j.zeros_like(y).at[:,:self.d,:].set(rt)
            db=bc*z+br*zr;dw=irt@(y.swapaxes(1,2)@db)
            mean=mean+(self.mu-.5*self.ito*j.diagonal(sig,axis1=1,axis2=2))*self.dt+self.rho*j.einsum('nij,nj->ni',rt,j.diagonal(dw,axis1=1,axis2=2))
            cov=cov+(1-self.rho**2)*sig*self.dt
            return (self.a*y+ou_sd*z@self.q,mean,cov),None
        (_,mean,cov),_=lax.scan(step,(y,j.zeros((n_predictive,self.d)),j.zeros((n_predictive,self.d,self.d))),(z,zr))
        return mean,cov

    def _risk(self,mean,cov,weights,alphas):
        # weights shape portfolios x d; resulting columns portfolio x alpha.
        loc=-mean@weights.T;sd=j.sqrt(j.einsum('pi,nij,pj->np',weights,cov,weights))
        loc=loc[:,:,None];sd=sd[:,:,None];al=alphas[None,:]
        low=j.min(loc-12*sd,axis=0)+j.zeros_like(al);high=j.max(loc+12*sd,axis=0)+j.zeros_like(al)
        def step(_,bounds):
            lo,hi=bounds;mid=(lo+hi)/2;tail=ndtr((loc-mid)/sd).mean(0)
            return j.where(tail>al,mid,lo),j.where(tail>al,hi,mid)
        lo,hi=lax.fori_loop(0,55,step,(low,high));q=(lo+hi)/2;zz=(q-loc)/sd
        es=(loc*ndtr(-zz)+sd*j.exp(-.5*zz**2)/j.sqrt(2*j.pi)).mean(0)/al
        return q,es
