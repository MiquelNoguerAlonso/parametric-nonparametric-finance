"""Coupled OU-factor grids and Gaussian-mixture weekly risk.

Exact OU endpoints are shared by nested grids. Only the return-integral
left-point rule differs. Both the OU integral and Brownian return driver are
aggregated, preserving their exact joint Gaussian covariance.
"""
import numpy as np
from wishart_tempered import TemperedWishart,roots,j,jax,jr,lax

class CoupledWishart(TemperedWishart):
    def __init__(self,par,finest=32,grids=(4,8,16,32),days=5):
        super().__init__(par,finest)
        self.grids=tuple(grids);self.days=days
        if any(finest%s for s in grids):raise ValueError('Grids must divide finest grid')
        self.coupled=jax.jit(self._coupled,static_argnums=(1,))
    def _coupled(self,key,n):
        ky,k1,k2=jr.split(key,3);y0=self._initial(ky,n)
        z=jr.normal(k1,(self.days*self.substeps,n,self.beta,self.d),dtype=j.float64)
        e=jr.normal(k2,z.shape,dtype=j.float64)
        ou=j.sqrt(self.ouv)*z
        db=self.cross/j.sqrt(self.ouv)*z+j.sqrt(self.dt-self.cross**2/self.ouv)*e
        out=[]
        for s in self.grids:
            ratio=self.substeps//s;dt=1./s;steps=self.days*s
            b=db.reshape(steps,ratio,n,self.beta,self.d).sum(1)
            weights=j.exp(-self.k*self.dt*j.arange(ratio-1,-1,-1)/2)
            u=(ou.reshape(steps,ratio,n,self.beta,self.d)*weights[None,:,None,None,None]).sum(1)
            decay=j.exp(-self.k*dt/2)
            def step(carry,noise):
                y,mean,cov=carry;delta,ouint=noise
                sig=y.swapaxes(1,2)@y;rt,irt=roots(sig);dw=irt@(y.swapaxes(1,2)@delta)
                mean+=(self.mu-.5*self.ito*j.diagonal(sig,axis1=1,axis2=2))*dt+self.rho*j.einsum('nij,nj->ni',rt,j.diagonal(dw,axis1=1,axis2=2))
                cov+=(1-self.rho**2)*sig*dt
                return (decay*y+ouint@self.q,mean,cov),None
            (end,mean,cov),_=lax.scan(step,(y0,j.zeros((n,self.d)),j.zeros((n,self.d,self.d))),(b,u))
            out.append((end,mean,cov))
        return out
