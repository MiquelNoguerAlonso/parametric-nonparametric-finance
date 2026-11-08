"""Numerical falsification checks for formulas and implementation contracts."""
import json,tempfile
from pathlib import Path
import numpy as np
import scipy.stats as st
from scipy.special import gammaln
from scipy.integrate import quad
import jax,jax.numpy as jnp
from models import Neural,MetaT,noise,log_t,tmap
from experiment import es,ewma

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,condition,detail):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})
    if not condition:raise AssertionError(name+': '+str(detail))

def main():
    r=np.random.default_rng(7)
    for nu in [3.,4.,8.]:
        k=np.exp(gammaln((nu+1)/2)+(nu/2-1)*np.log(nu)-.5*np.log(np.pi)-gammaln(nu/2))
        numerical=st.t.sf(1e5,nu)*1e5**nu
        check(f'Student survival constant nu={nu}',abs(numerical/k-1)<1e-7,{'formula':k,'numerical':numerical})
    for ni in [3.,8.]:
        k=lambda n:np.exp(gammaln((n+1)/2)+(n/2-1)*np.log(n)-.5*np.log(np.pi)-gammaln(n/2))
        y=1e6;ratio=tmap(y,4.,ni)/((k(ni)/k(4.))**(1/ni)*y**(4./ni))
        check(f'Meta-t coordinate asymptotic nu={ni}',abs(ratio-1)<2e-6,float(ratio))
    X=r.normal(size=(100,5))
    lp=np.asarray(log_t(jnp.asarray(X),4.));target=st.multivariate_t.logpdf(X,shape=np.eye(5),df=4)
    check('Elliptical t log density versus scipy',np.max(np.abs(lp-target))<1e-5,float(np.max(np.abs(lp-target))))
    for base in ['t-elliptical','t-independent']:
        x=np.asarray(noise(jax.random.PRNGKey(19),300000,2,base,3.));cut=np.quantile(x,.99,axis=0)
        co=float(np.mean(x[x[:,0]>cut[0],1]>cut[1]))
        check('Shared versus independent tail dependence: '+base,co>.08 if base=='t-elliptical' else co<.025,co)
    slope=-np.log((st.t.sf(2000,3)/st.t.sf(1000,3))**2)/np.log(2)
    check('Independent-noise ReLU minimum has index six',abs(slope-6)<1e-4,float(slope))
    # Exact parametric density identity and forward/inverse Jacobian contract.
    m=MetaT();m.marg=np.array([[3.,.2,1.1],[8.,-.1,.7]]);m.P=np.array([[1.,.5],[.5,1.]]);m.nu=4.;m.d=2
    x=m.sample(20000,123);y=m.to_y(x);back=m.from_y(y)
    check('Meta-t round trip',np.max(np.abs(x-back))<1e-7,float(np.max(np.abs(x-back))))
    mean_u=st.t.cdf((x-m.marg[:,1])/m.marg[:,2],m.marg[:,0]).mean(0)
    check('Exact meta-t uniform marginal PIT',np.max(np.abs(mean_u-.5))<.01,mean_u.tolist())
    n=Neural('flow',2,steps=250);n.fit(r.normal(size=(600,2)),r.normal(size=(200,2)))
    xx=jnp.array([[.2,-.4],[2.,1.]])
    z,ld=n.flow_forward(n.p,xx,True);xxx,_=n.flow_forward(n.p,z)
    check('Flow inverse round trip',np.max(np.abs(np.asarray(xxx-xx)))<1e-5,float(np.max(np.abs(np.asarray(xxx-xx)))))
    jac=jax.jacfwd(lambda u:n.flow_forward(n.p,u[None],True)[0][0])(xx[0])
    check('Flow analytical log Jacobian',abs(float(jnp.linalg.slogdet(jac)[1]-ld[0]))<1e-5,float(jnp.linalg.slogdet(jac)[1]-ld[0]))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'fit.npz';n.save(path);loaded=Neural.load(path)
        check('Checkpoint round trip',np.allclose(n.sample(1000),loaded.sample(1000)),True)
    a=.01;epsilon=.07;u=(np.arange(100000)+.5)/100000;p=st.norm.ppf(u);q=p+(u>1-a)*epsilon/a
    w1=st.wasserstein_distance(p,q);de=es(q,a)-es(p,a)
    check('Sharp Wasserstein-ES construction',abs(w1-epsilon)<1e-10 and abs(de-epsilon/a)<1e-10,{'w1':w1,'delta_es':de})
    # Tail integral of the exact quantile equals the closed-form Student ES.
    q=st.t.ppf(a,3);formula=np.sqrt(.75)*st.t.pdf(q,3)/a*(3+q*q)/2
    integ=quad(lambda u:-np.sqrt(.75)*st.t.ppf(u,3),0,a,epsabs=1e-8)[0]/a
    check('Student-t portfolio ES integral',abs(formula-integ)<1e-6,{'formula':formula,'integral':integ})
    # The common CRSP volatility filter must be predictable.
    xx=r.normal(size=(1000,5));changed=xx.copy();changed[700:]*=100
    check('EWMA no future leakage',np.array_equal(ewma(xx)[:701],ewma(changed)[:701]),True)
    # FZ0 population score stationarity at the Gaussian VaR/ES pair.
    v=st.norm.ppf(1-a);s=st.norm.pdf(v)/a
    loss=lambda vv,ss:np.log(ss)+vv/ss-1+(st.norm.pdf(vv)-vv*st.norm.sf(vv))/(a*ss)
    opt=loss(v,s);others=[loss(v*.9,s),loss(v*1.1,s),loss(v,s*.9),loss(v,s*1.1)]
    check('FZ0 expected-score minimum locally',all(o>opt for o in others),{'optimum':opt,'perturbations':others})
    (ROOT/'results'/'mathematical_checks.json').write_text(json.dumps(checks,indent=2))
    print(f'{len(checks)} checks passed')

if __name__=='__main__':main()
