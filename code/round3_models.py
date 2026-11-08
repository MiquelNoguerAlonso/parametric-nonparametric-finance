"""Dense-output control; preserves the retained Neural implementation by default."""
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from models import Neural

class DenseFlow(Neural):
    def initialize_flow_output(self,p,x):
        cov=np.cov(np.asarray(x,dtype=float),rowvar=False)+1e-6*np.eye(self.d)
        chol=np.linalg.cholesky(cov);diag=np.diag(chol)
        p.update(mix_lower=jnp.asarray((chol/diag[None,:])[np.tril_indices(self.d,-1)]),
                 mix_upper=jnp.zeros(self.d*(self.d-1)//2),mix_logdiag=jnp.asarray(np.log(diag)))
        return p

    def factors(self,p):
        lower=jnp.eye(self.d).at[np.tril_indices(self.d,-1)].set(p['mix_lower'])
        upper=jnp.diag(jnp.exp(p['mix_logdiag'])).at[np.triu_indices(self.d,1)].set(p['mix_upper'])
        return lower,upper

    def flow_forward(self,p,z,inverse=False):
        lower,upper=self.factors(p)
        if inverse:
            v=solve_triangular(lower,z.T,lower=True,unit_diagonal=True)
            v=solve_triangular(upper,v,lower=False).T
            x,ld=super().flow_forward(p,v,True)
            return x,ld-p['mix_logdiag'].sum()
        x,ld=super().flow_forward(p,z,False)
        return x@upper.T@lower.T,ld+p['mix_logdiag'].sum()
