"""Independent checks for the new dense map and predictable EWMA recursion."""
from pathlib import Path
import json
import numpy as np
import scipy.stats as st
import jax,jax.numpy as jnp
from models import mlp_init
from round3_models import DenseFlow
from round3_controls import ewma_state

def main():
    n=DenseFlow('flow',2,1,base='gaussian')
    p={'a':jnp.zeros(2),'b':jnp.zeros(2),'nu_raw':jnp.array(3.),
       'layers':[[(jnp.zeros_like(w),jnp.zeros_like(b)) for w,b in mlp_init(jax.random.PRNGKey(i),[len(c),32,32,2*len(d)])] for i,(c,d) in enumerate(n.idx)],
       'mix_lower':jnp.array([.7]),'mix_upper':jnp.array([.2]),'mix_logdiag':jnp.log(jnp.array([1.1,.8]))}
    l,u=n.factors(p);A=np.asarray(l@u);z=jnp.asarray(np.random.default_rng(13).normal(size=(100,2)))
    x,ld=n.flow_forward(p,z);back,ild=n.flow_forward(p,x,True)
    inverse=float(np.max(abs(np.asarray(back-z))));jac=np.asarray(jax.jacfwd(lambda a:n.flow_forward(p,a[None])[0][0])(z[0]))
    logdet=float(abs(np.linalg.slogdet(jac)[1]-np.asarray(ld)[0]))
    likelihood=float(np.max(abs(np.asarray(n.flow_logpdf(p,x))-st.multivariate_normal.logpdf(np.asarray(x),cov=A@A.T))))
    assert inverse<2e-6 and logdet<1e-6 and likelihood<5e-6
    y=np.random.default_rng(4).normal(size=400);changed=y.copy();changed[250:]+=100
    before=ewma_state(y,.94,1.);after=ewma_state(changed,.94,1.)
    assert np.array_equal(before[:251],after[:251])
    checks=dict(passed=True,roundtrip_max_error=inverse,independent_jacobian_logdet_error=logdet,
        gaussian_embedding_logpdf_error=likelihood,ewma_future_perturbation_predictability=True,
        frozen_wishart_excess=6/6,heston_stationary_shape=2*3*.04/.5**2,heston_frozen_excess=3/(2*3*.04/.5**2),
        equivalent_wishart_shape=4*3*.04/.5**2,target_kappa_per_day=3/252)
    root=Path(__file__).resolve().parents[1];(root/'referee_round3/analysis/kernel_checks.json').write_text(json.dumps(checks,indent=2));print(checks)

if __name__=='__main__':main()
