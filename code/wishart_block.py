"""Overlapping block particle updates of the unchanged Wishart model.

At date t the base cloud precedes the first observation in the retained block.
The joint block likelihood is used exactly once. It updates earlier covariance
increments continuously and produces the filtering state after t, not a forecast
that has seen t. Reported normalizer is a block normalizer, not a daily NLL.
"""
import numpy as np
from wishart_tempered import TemperedWishart,j,jr,lax,logpdf,logsumexp,resample

class BlockWishart(TemperedWishart):
    def __init__(self,par,substeps=8,lag=2,**kwargs):
        if kwargs.get('defensive',0.):raise ValueError('Block implementation uses the auxiliary proposal')
        self.lag=lag
        super().__init__(par,substeps=substeps,**kwargs)

    def _block_noise(self,key,n):
        k1,k2=jr.split(key)
        return (jr.normal(k1,(n,self.lag,2,self.substeps,self.beta,self.d),dtype=j.float64),
                jr.normal(k2,(n,self.lag,self.d),dtype=j.float64))

    def _block_proposal(self,start,obs,z,eps):
        lr=j.zeros(start.shape[0]);y=start
        for day in range(self.lag):
            if day:
                sig=y.swapaxes(1,2)@y;sig=self.c+self.average_decay*(sig-self.c) if self.moment_auxiliary else sig;drift=self.mu-.5*self.ito*j.diagonal(sig,axis1=1,axis2=2)
                lr=lr+logpdf(obs[day],drift,sig)
            y,ratio=self._proposal(y,obs[day],z[:,day],eps[:,day]);lr=lr+ratio
        return y,lr

    def _update(self,cloud,observed,key):
        n=cloud.shape[0];sig=cloud.swapaxes(1,2)@cloud
        sig=self.c+self.average_decay*(sig-self.c) if self.moment_auxiliary else sig
        drift=self.mu-.5*self.ito*j.diagonal(sig,axis1=1,axis2=2)
        aux=logpdf(observed[0],drift,sig);normaux=logsumexp(self.aux_power*aux);ap=j.exp(self.aux_power*aux-normaux);pess=1/j.sum(ap**2)
        key,k1,k2=jr.split(key,3);idx=resample(ap,k1);z,eps=self._block_noise(k2,n)
        def evaluate(idx,z,eps):
            end,ratio=self._block_proposal(cloud[idx],observed,z,eps)
            return end,ratio+(1-self.aux_power)*aux[idx]
        end,lr=evaluate(idx,z,eps);rawess=1/j.sum(j.exp(lr-logsumexp(lr))**2)
        state=(j.array(0.),idx,z,eps,end,lr,key,j.array(0),j.array(0.),j.array(float(n)),j.array(0.),j.array(0.))
        def cond(s):return (s[0]<1.-1e-12)&(s[7]<self.max_stages)
        def body(s):
            gamma,idx,z,eps,end,lr,key,st,logz,miness,acc,attempt=s
            def ess(delta):
                w=j.exp(delta*lr-logsumexp(delta*lr));return 1/j.sum(w*w)
            def bisect(_,bounds):
                lo,hi=bounds;mid=(lo+hi)/2;ok=ess(mid)>=self.ess_fraction*n
                return j.where(ok,mid,lo),j.where(ok,hi,mid)
            lo,_=lax.fori_loop(0,32,bisect,(j.array(0.),1.-gamma))
            delta=j.where(ess(1.-gamma)>=self.ess_fraction*n,1.-gamma,lo)
            gamma=gamma+delta;lw=delta*lr;logz+=logsumexp(lw)-j.log(n)
            prob=j.exp(lw-logsumexp(lw));miness=j.minimum(miness,1/j.sum(prob**2))
            key,kr=jr.split(key);sel=resample(prob,kr);idx,z,eps,end,lr=idx[sel],z[sel],eps[sel],end[sel],lr[sel]
            for scale in self.scales:
                key,kn,ku=jr.split(key,3);zn,en=self._block_noise(kn,n)
                zp=j.sqrt(1-scale**2)*z+scale*zn;ep=j.sqrt(1-scale**2)*eps+scale*en
                yp,lp=evaluate(idx,zp,ep);accept=j.log(jr.uniform(ku,(n,),dtype=j.float64))<gamma*(lp-lr)
                z=j.where(accept[:,None,None,None,None,None],zp,z);eps=j.where(accept[:,None,None],ep,eps)
                end=j.where(accept[:,None,None],yp,end);lr=j.where(accept,lp,lr);acc+=accept.sum();attempt+=n
            key,ki,ku=jr.split(key,3);ip=j.searchsorted(j.cumsum(ap).at[-1].set(1.),jr.uniform(ki,(n,),dtype=j.float64),side='right')
            yp,lp=evaluate(ip,z,eps);accept=j.log(jr.uniform(ku,(n,),dtype=j.float64))<gamma*(lp-lr)
            idx=j.where(accept,ip,idx);end=j.where(accept[:,None,None],yp,end);lr=j.where(accept,lp,lr);acc+=accept.sum();attempt+=n
            return gamma,idx,z,eps,end,lr,key,st+1,logz,miness,acc,attempt
        gamma,idx,z,eps,end,lr,key,st,logz,miness,acc,attempt=lax.while_loop(cond,body,state)
        return end,j.stack([gamma,st,miness,pess,rawess,acc/j.maximum(attempt,1),-normaux+j.log(n)-logz])
