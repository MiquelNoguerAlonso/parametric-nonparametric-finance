"""Independent analytic/kernel checks for the new Wishart implementation."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from wishart_model import initial_factors,advance,sample_windows,mixture_risk,filter_series,guided_update

ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.time();rng=np.random.default_rng(830019);d=5
    c=.5*np.eye(d)+.5*np.ones((d,d));p=dict(C=c.tolist(),mu=[0.]*d,beta=6,kappa=.05,rho=-.7,ito_scale=0.)
    y=initial_factors(p,100000,rng);s=y.transpose(0,2,1)@y
    mean_error=float(np.max(abs(s.mean(0)-c)))
    theoretical_var=2/6
    variance_error=float(abs(s[:,0,0].var()-theoretical_var))
    cross_error=float(abs(np.cov(s[:,0,0],s[:,1,1])[0,1]-2*c[0,1]**2/6))
    assert mean_error<.02 and variance_error<.02 and cross_error<.015
    assert np.linalg.eigvalsh(s).min()>0
    y1,m,v,r=advance(y,p,rng,4,True);s1=y1.transpose(0,2,1)@y1
    correlation=np.corrcoef(s[:,0,0],s1[:,0,0])[0,1]
    assert abs(correlation-np.exp(-p['kappa']))<.01
    assert np.max(abs(np.cov(r,rowvar=False)-c))<.025
    corr0=s[:,0,1]/np.sqrt(s[:,0,0]*s[:,1,1]);corr1=s1[:,0,1]/np.sqrt(s1[:,0,0]*s1[:,1,1])
    assert np.mean((corr1-corr0)**2)>.001
    # Conditional Gaussian mixture risk reduces to the analytic normal case.
    for alpha in [.01,.025,.001]:
        q,e=mixture_risk(np.zeros((1,5)),c[None,:,:],np.ones(5)/5,alpha)
        sd=np.sqrt(np.mean(c));z=norm.isf(alpha)
        assert abs(q-sd*z)<1e-8 and abs(e-sd*norm.pdf(z)/alpha)<1e-8
    # Scalar stationary variance is Gamma(beta/2, 2*C/beta).
    scalar=dict(C=[[.7]],mu=[0.],beta=4,kappa=.1,rho=0.,ito_scale=0.)
    ys=initial_factors(scalar,100000,rng);vs=(ys*ys).sum((1,2))
    assert abs(vs.mean()-.7)<.01 and abs(vs.var()-2*.7**2/4)<.01
    # A zero-leverage, nearly frozen stationary covariance has known CF.
    frozen=dict(p,kappa=1e-10,rho=0.)
    xx=sample_windows(frozen,100000,830020,days=1,substeps=1)[:,:,0]
    u=np.array([.2,-.1,.3,0.,.2]);cf=np.exp(1j*(xx@u)).mean();true=(1+u@c@u/6)**(-3)
    assert abs(cf-true)<.006
    # Predictable forecasts: modifying day 35 changes no forecast through day 35.
    returns=r[:50];changed=returns.copy();changed[35]+=5
    kw=dict(seed=830021,n_particles=128,n_predictive=256,substeps=2,forecast_from=30,weights={'equal':np.ones(5)/5})
    a,_=filter_series(p,returns,**kw);b,_=filter_series(p,changed,**kw)
    assert a[:6]==b[:6] and a[6:]!=b[6:]
    # With one return substep, frozen/actual observations coincide exactly.
    yg=initial_factors(p,512,rng)
    _,guided=guided_update(yg,p,np.full(5,-3.),rng,substeps=1)
    assert guided['log_correction_range']<1e-8
    assert abs(guided['ess']-512)<1e-7
    # Model leverage responds with the correct sign in daily/next-day squares.
    w=sample_windows(p,100000,830022,days=2,substeps=4)
    leverage=float(np.mean(w[:,:,0]*(w[:,:,1]**2-1)))
    assert leverage<-.01
    result=dict(passed=True,stationary_mean_max_error=mean_error,stationary_variance_error=variance_error,
        stationary_cross_covariance_error=cross_error,covariance_autocorrelation=float(correlation),
        expected_covariance_autocorrelation=float(np.exp(-p['kappa'])),
        correlation_increment_mean_square=float(np.mean((corr1-corr0)**2)),
        frozen_characteristic_function_error=float(abs(cf-true)),leverage_moment=leverage,
        scalar_gamma_check=True,gaussian_risk_check=True,forecast_no_lookahead_check=True,
        fully_adapted_one_step_check=True,
        no_eigenvalue_projection_used=True,seconds=time.time()-start)
    (ROOT/'wishart_extension/analysis/kernel_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
