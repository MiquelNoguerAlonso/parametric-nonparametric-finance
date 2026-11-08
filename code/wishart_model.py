"""Restricted, full-matrix Heston--Wishart return model.

The covariance SDE is the one in Noguer i Alonso (2025), Section 4:
M=-kappa I/2, Q=sqrt(kappa/beta) C^(1/2), beta an integer >= d+1.
Diagonal leverage uses the *symmetric* covariance square root. Covariance
transitions are exact OU-factor transitions; returns use left-point integration.
No printed Riccati/FFT formula from that preprint is used.
"""
import itertools
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.optimize import brentq


def symmetric_root(a, inverse=False):
    v, u = np.linalg.eigh(a)
    if np.any(v <= 0):
        raise FloatingPointError('Nonpositive covariance eigenvalue')
    return (u * (1 / np.sqrt(v) if inverse else np.sqrt(v))[..., None, :]) @ np.swapaxes(u, -1, -2)


def initial_factors(par, n, rng):
    c = np.asarray(par['C'])
    return rng.standard_normal((n, par['beta'], len(c))) @ symmetric_root(c) / np.sqrt(par['beta'])


def advance(y, par, rng, substeps=4, draw_return=True):
    """One day, conditional Gaussian mean/covariance and optional return draw.

    Normal drivers of the exact OU integral and the Brownian increment are
    jointly sampled, with their exact covariance. This preserves leverage
    as the left-point return integration is refined.
    """
    n, beta, d = y.shape
    dt = 1. / substeps
    k, rho = par['kappa'], par['rho']
    a = np.exp(-k * dt / 2)
    ou_sd = np.sqrt(-np.expm1(-k * dt) / k)
    cross = -np.expm1(-k * dt / 2) / (k / 2)
    brown_coeff = cross / ou_sd
    brown_resid = np.sqrt(max(0., dt - brown_coeff ** 2))
    q = np.sqrt(k / beta) * symmetric_root(np.asarray(par['C']))
    mu = np.asarray(par['mu'])
    ito = par.get('ito_scale', 0.)
    mean = np.zeros((n, d))
    cov = np.zeros((n, d, d))
    x = np.zeros((n, d)) if draw_return else None
    for _ in range(substeps):
        sigma = np.swapaxes(y, 1, 2) @ y
        ev, u = np.linalg.eigh(sigma)
        if np.any(ev <= 0):
            raise FloatingPointError('OU factors lost rank')
        root = (u * np.sqrt(ev)[:, None, :]) @ np.swapaxes(u, 1, 2)
        invroot = (u / np.sqrt(ev)[:, None, :]) @ np.swapaxes(u, 1, 2)
        z = rng.standard_normal(y.shape)
        db = brown_coeff * z + brown_resid * rng.standard_normal(y.shape)
        dw = invroot @ (np.swapaxes(y, 1, 2) @ db)
        lever = rho * np.einsum('nij,nj->ni', root, np.diagonal(dw, axis1=1, axis2=2))
        drift = (mu - .5 * ito * np.diagonal(sigma, axis1=1, axis2=2)) * dt
        mean += drift + lever
        cov += (1 - rho ** 2) * sigma * dt
        if draw_return:
            x += drift + lever + np.sqrt((1 - rho ** 2) * dt) * np.einsum('nij,nj->ni', root, rng.standard_normal((n, d)))
        y = a * y + ou_sd * (z @ q)
    return y, mean, cov, x


def sample_windows(par, n, seed, days=5, substeps=4):
    rng = np.random.default_rng(seed)
    y = initial_factors(par, n, rng)
    out = []
    for _ in range(days):
        y, _, _, r = advance(y, par, rng, substeps, True)
        out.append(r)
    return np.stack(out, axis=2)


def features(windows, c, daily_mean):
    """Pooled marginal kurtosis, four squared-return lags, four leverage lags,
    and cross-asset squared-return covariance. No future/test observations.
    """
    z = (windows - daily_mean[None, :, None]) / np.sqrt(np.diag(c))[None, :, None]
    square = z*z - 1
    result = [float(np.mean(z**4) - 3)]
    result += [float(np.mean(square[:, :, :-lag] * square[:, :, lag:])) for lag in range(1, 5)]
    result += [float(np.mean(z[:, :, :-lag] * square[:, :, lag:])) for lag in range(1, 5)]
    i, j = np.triu_indices(len(c), 1)
    result.append(float(np.mean(square[:, i, :] * square[:, j, :])))
    return np.asarray(result)


def fit_smm(windows, seed, ito_scale=0., fit_scenarios=10000, substeps=4, daily=None):
    """Two-stage, fixed-grid simulated method of moments, not MLE.

    Sample mean/full covariance use training only. The ten moment scales and
    parameter grid are fixed in the protocol before running any new scores.
    """
    n, d, _ = windows.shape
    daily = windows.transpose(0, 2, 1).reshape(-1, d) if daily is None else np.asarray(daily)
    daily_mean = daily.mean(0)
    c = np.cov(daily, rowvar=False)
    c = .99*c + .01*np.diag(np.diag(c))
    mu = daily_mean + .5*ito_scale*np.diag(c)
    observed = features(windows, c, daily_mean)
    scale = np.array([3., 1., 1., 1., 1., .1, .1, .1, .1, 1.])
    candidates = []
    # Identical seeds within beta give common random drivers across kappa/rho.
    for beta, kappa, rho in itertools.product([d+1, 2*d, 4*d], [.002, .01, .05, .2], [-.7, 0., .7]):
        par = dict(C=c.tolist(), mu=mu.tolist(), beta=beta, kappa=kappa, rho=rho, ito_scale=ito_scale)
        sim = sample_windows(par, fit_scenarios, seed+100*beta, substeps=substeps)
        moments = features(sim, c, daily_mean)
        objective = float(np.mean(((moments-observed)/scale)**2))
        candidates.append(dict(beta=beta, kappa=kappa, rho=rho, objective=objective, simulated_moments=moments.tolist()))
    best = min(candidates, key=lambda x:x['objective'])
    par = dict(C=c.tolist(), mu=mu.tolist(), beta=best['beta'], kappa=best['kappa'], rho=best['rho'], ito_scale=ito_scale)
    return par, dict(method='fixed-grid simulated method of moments', candidates=candidates,
        observed_moments=observed.tolist(), moment_scales=scale.tolist(), objective=best['objective'],
        n_windows=n, n_days_in_windows=5*n, fit_scenarios=fit_scenarios, fit_seed=seed,
        substeps=substeps, covariance_shrinkage=.01, kappa_time_unit='one trading day')


def normal_logpdf(x, mean, cov):
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, (np.asarray(x)-mean)[..., None])[..., 0]
    d = mean.shape[-1]
    return -.5*(d*np.log(2*np.pi) + 2*np.log(np.diagonal(chol,axis1=-2,axis2=-1)).sum(-1) + (z*z).sum(-1))


def systematic(prob, n, rng):
    u = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(prob)
    cumulative[-1] = 1.
    return np.searchsorted(cumulative, u, side='right')


def mixture_risk(mean, cov, w, alpha):
    """Rao--Blackwellized univariate loss quantile/ES, no return resampling."""
    loc = -mean @ w
    sd = np.sqrt(np.einsum('i,nij,j->n', w, cov, w))
    objective = lambda q: float(ndtr((loc-q)/sd).mean()-alpha)
    lo, hi = float(np.min(loc-12*sd)), float(np.max(loc+12*sd))
    q = brentq(objective, lo, hi, xtol=1e-10)
    z = (q-loc)/sd
    e = float(np.mean(loc*ndtr(-z)+sd*np.exp(-.5*z*z)/np.sqrt(2*np.pi))/alpha)
    if not np.isfinite(e) or e < q or e <= 0:raise FloatingPointError((q,e))
    return float(q), e


def guided_update(y, par, observed, rng, substeps=4):
    """Auxiliary PF, with an observation-guided Brownian proposal.

    The frozen-covariance observation law selects ancestor states and gives a
    Gaussian conditional proposal for all factor-Brownian increments. The
    actual evolving-covariance conditional likelihood corrects that proposal.
    For one return substep the correction is exactly one (fully adapted).
    Observations are used here only AFTER issuing that date's forecast.
    """
    n, beta, d = y.shape;dt=1./substeps;k=par['kappa'];rho=par['rho']
    sigma0=y.transpose(0,2,1)@y
    drift0=np.asarray(par['mu'])-.5*par.get('ito_scale',0.)*np.diagonal(sigma0,axis1=1,axis2=2)
    auxiliary=normal_logpdf(observed,drift0,sigma0)
    normaux=logsumexp(auxiliary);ancprob=np.exp(auxiliary-normaux)
    ancestor=systematic(ancprob,n,rng)
    y=y[ancestor].copy();sigma0=sigma0[ancestor];drift0=drift0[ancestor]
    ev,u=np.linalg.eigh(sigma0)
    root0=(u*np.sqrt(ev)[:,None,:])@u.transpose(0,2,1)
    invroot0=(u/np.sqrt(ev)[:,None,:])@u.transpose(0,2,1)
    projection=invroot0@y.transpose(0,2,1)
    db=rng.standard_normal((n,substeps,beta,d))*np.sqrt(dt)
    frozen_dw=projection@db.sum(1)
    pseudo=rho*np.diagonal(frozen_dw,axis1=1,axis2=2)+np.sqrt(1-rho*rho)*rng.standard_normal((n,d))
    target=np.einsum('nij,nj->ni',invroot0,np.asarray(observed)-drift0)
    # Gaussian conditional simulation; Cov(db_j, pseudo)=dt*rho*A^T.
    shift=dt*rho*projection.transpose(0,2,1)*(target-pseudo)[:,None,:]
    db=db+shift[:,None,:,:]
    frozen_dw=projection@db.sum(1)
    frozen_mean=drift0+rho*np.einsum('nij,nj->ni',root0,np.diagonal(frozen_dw,axis1=1,axis2=2))
    log_frozen=normal_logpdf(observed,frozen_mean,(1-rho*rho)*sigma0)
    a=np.exp(-k*dt/2);ou_variance=-np.expm1(-k*dt)/k
    cross=-np.expm1(-k*dt/2)/(k/2)
    residual=np.sqrt(max(0.,ou_variance-cross*cross/dt))
    q=np.sqrt(k/beta)*symmetric_root(np.asarray(par['C']))
    mean=np.zeros((n,d));cov=np.zeros((n,d,d))
    for j in range(substeps):
        sigma=y.transpose(0,2,1)@y;ev,u=np.linalg.eigh(sigma)
        if np.any(ev<=0):raise FloatingPointError('Guided OU factors lost rank')
        root=(u*np.sqrt(ev)[:,None,:])@u.transpose(0,2,1)
        invroot=(u/np.sqrt(ev)[:,None,:])@u.transpose(0,2,1)
        dw=invroot@(y.transpose(0,2,1)@db[:,j])
        drift=(np.asarray(par['mu'])-.5*par.get('ito_scale',0.)*np.diagonal(sigma,axis1=1,axis2=2))*dt
        mean+=drift+rho*np.einsum('nij,nj->ni',root,np.diagonal(dw,axis1=1,axis2=2))
        cov+=(1-rho*rho)*sigma*dt
        integral=(cross/dt)*db[:,j]+residual*rng.standard_normal(y.shape)
        y=a*y+integral@q
    log_ratio=normal_logpdf(observed,mean,cov)-log_frozen
    correction_norm=logsumexp(log_ratio);weights=np.exp(log_ratio-correction_norm)
    indices=systematic(weights,n,rng)
    return y[indices].copy(),dict(ess=float(1/(weights@weights)),parent_ess=float(1/(ancprob@ancprob)),
        nll=float(-normaux+np.log(n)-correction_norm+np.log(n)),
        log_correction_range=float(np.ptp(log_ratio)),proposals=n)


def filter_series(par, returns, seed, n_particles=512, n_predictive=4096,
                  substeps=4, forecast_from=None, weights=None, callback=None):
    """Auxiliary covariance filter, using each return only after its forecast.

    Integrates the independent return Brownian motion analytically. Covariance
    paths give a finite Gaussian mixture. The filter retains n_particles
    resampled OU states. Diagnostics describe particle approximation error.
    """
    rng = np.random.default_rng(seed)
    y = initial_factors(par, n_particles, rng)
    predictions, diagnostics = [], []
    for t, r in enumerate(returns):
        predict = forecast_from is not None and t >= forecast_from
        if predict:
            idx = rng.integers(n_particles, size=n_predictive)
            start = y[idx]
            _, mean, cov, _ = advance(start, par, rng, substeps, False)
        # Forecast is formed before r is used below.
        if predict:
            forecasts = {(name, a):mixture_risk(mean,cov,np.asarray(w),a)
                         for name,w in weights.items() for a in [.01,.025]}
            predictions.append(dict(t=t, forecasts=forecasts))
        y, diagnostic=guided_update(y,par,r,rng,substeps)
        diagnostics.append(dict(t=t,**diagnostic))
        if callback is not None and (t+1)%500==0:callback(t+1)
    return predictions, diagnostics
